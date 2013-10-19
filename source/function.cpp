// Petter Strandmark 2012-2013.

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <typeinfo>

#ifdef USE_OPENMP
	#include <omp.h>
#endif

#include <spii/function.h>
#include <spii/spii.h>

namespace spii {

// These two structs are used by Function to store added
// variables and terms.
struct AddedVariable
{
	int user_dimension;
	int solver_dimension;
	size_t global_index;
	bool is_constant;
	std::shared_ptr<ChangeOfVariables> change_of_variables;
	mutable std::vector<double>  temp_space;
};

struct AddedTerm
{
	std::shared_ptr<const Term> term;
	std::vector<AddedVariable*> user_variables;
	// Temporary storage for a point and hessian.
	mutable std::vector<double*> temp_variables;
	mutable std::vector< std::vector<Eigen::MatrixXd> > hessian;
};

class Function::Implementation
{
public:
	Implementation(Function* function_interface);

	// Implemenations of functions in the public interface.
	double evaluate(const Eigen::VectorXd& x,
	                Eigen::VectorXd* gradient,
	                Eigen::MatrixXd* hessian) const;
	double evaluate(const Eigen::VectorXd& x,
	                Eigen::VectorXd* gradient,
	                Eigen::SparseMatrix<double>* hessian) const;
	Interval<double> evaluate(const std::vector<Interval<double>>& x) const;

	// Adds a variable to the function. All variables must be added
	// before any terms containing them are added.
	void add_variable_internal(double* variable,
	                           int dimension,
	                           std::shared_ptr<ChangeOfVariables> change_of_variables = 0);

	void set_constant(double* variable, bool is_constant);

	// Copies variables from a global vector x to the Function's
	// local storage.
	void copy_global_to_local(const Eigen::VectorXd& x) const;
	// Copies variables from a global vector x to the storage
	// provided by the user.
	void copy_global_to_user(const Eigen::VectorXd& x) const;
	// Copies variables from a the storage provided by the user
	// to a global vector x.
	void copy_user_to_global(Eigen::VectorXd* x) const;
	// Copies variables from a the storage provided by the user
	// to the Function's local storage.
	void copy_user_to_local() const;

	// Evaluates the function at the point in the local storage.
	double evaluate_from_local_storage() const;

	// Clears the function to the empty function.
	void clear();

	// All variables added to the function.
	std::map<double*, AddedVariable> variables;

	// Each variable can have several dimensions. This member
	// keeps track of the total number of scalars.
	size_t number_of_scalars;
	size_t number_of_constants;

	// Constant term (default: 0)
	double constant;

	// All terms added to the function.
	std::vector<AddedTerm> terms;

	// Number of threads used for evaluation.
	int number_of_threads;

	// Allocates temporary storage for gradient evaluations.
	// Should be called automatically at first evaluate()
	void allocate_local_storage() const;

	// If finalize has been called.
	mutable bool local_storage_allocated;
	// Has to be mutable because the temporary storage
	// needs to be written to.
	mutable std::vector< std::vector<Eigen::VectorXd> >
		thread_gradient_scratch;
	mutable std::vector<Eigen::VectorXd>
		thread_gradient_storage;

	// Stored how many element were used the last time the Hessian
	// was created.
	mutable size_t number_of_hessian_elements;

	Function* interface;
};

Function::Function() :
	impl(new Function::Implementation(this))
{ }

Function::Function(const Function& org)
	: impl(new Function::Implementation(this))
{
	*this = org;
}

Function::Implementation::Implementation(Function* function_interface) 
	: interface(function_interface)
{
	clear();
} 

Function::~Function()
{
	delete impl;
}

void Function::Implementation::clear()
{
	constant = 0;

	terms.clear();
	variables.clear();
	number_of_scalars = 0;
	number_of_constants = 0;

	thread_gradient_scratch.clear();
	thread_gradient_storage.clear();
	local_storage_allocated = false;

	number_of_hessian_elements = 0;

	#ifdef USE_OPENMP
		number_of_threads = omp_get_max_threads();
	#else
		number_of_threads = 1;
	#endif
}

Function& Function::operator = (const Function& org)
{
	if (this == &org) {
		return *this;
	}
	impl->clear();

	this->hessian_is_enabled = org.hessian_is_enabled;
	impl->constant = org.impl->constant;

	// TODO: respect global order.
	std::map<const AddedVariable*, double*> user_variables;
	for (const auto& added_variable: org.impl->variables) {
		impl->add_variable_internal(added_variable.first,
		                            added_variable.second.user_dimension,
		                            added_variable.second.change_of_variables);
		user_variables[&added_variable.second] = added_variable.first;
	}
	spii_assert(org.impl->variables.size() == user_variables.size());
	spii_assert(get_number_of_variables() == org.get_number_of_variables());
	spii_assert(get_number_of_scalars() == org.get_number_of_scalars());

	for (const auto& added_term: org.impl->terms) {
		std::vector<double*> vars;
		for (auto added_var: added_term.user_variables) {
			vars.push_back(user_variables[added_var]);
		}
		this->add_term(added_term.term, vars);
	}
	spii_assert(org.impl->variables.size() == user_variables.size());

	return *this;
}

Function& Function::operator += (const Function& org)
{
	impl->constant += org.impl->constant;

	// Check that there are no change of variables involved.
	for (const auto& added_variable: org.impl->variables) {
		spii_assert( ! added_variable.second.change_of_variables)
	}
	for (const auto& added_variable: impl->variables) {
		spii_assert( ! added_variable.second.change_of_variables)
	}

	// TODO: respect global order.
	std::map<const AddedVariable*, double*> user_variables;
	for (const auto& added_variable: org.impl->variables) {
		// No-op if the variable already exists.
		impl->add_variable_internal(added_variable.first,
		                            added_variable.second.user_dimension,
		                            added_variable.second.change_of_variables);
		user_variables[&added_variable.second] = added_variable.first;
	}

	for (auto const& added_term: org.impl->terms) {
		std::vector<double*> vars;
		for (auto added_var: added_term.user_variables) {
			vars.push_back(user_variables[added_var]);
		}
		this->add_term(added_term.term, vars);
	}

	return *this;
}

Function& Function::operator += (double constant_value)
{
	impl->constant += constant_value;
	return *this;
}

void Function::add_variable(double* variable,
                            int dimension)
{
	impl->add_variable_internal(variable, dimension);
}

size_t Function::get_variable_global_index(double* variable) const
{
	// Find the variable. This has to succeed.
	auto itr = impl->variables.find(variable);
	check(itr != impl->variables.end(), 
	      "Function::get_variable_global_index: variable not found.");

	return itr->second.global_index;
}

size_t Function::get_number_of_variables() const
{
	return impl->variables.size();
}

// Returns the current number of scalars the function contains.
// (each variable contains of one or several scalars.)
size_t Function::get_number_of_scalars() const
{
	return impl->number_of_scalars;
}

void Function::add_variable_internal(double* variable,
                                     int dimension,
                                     std::shared_ptr<ChangeOfVariables> change_of_variables)
{
	impl->add_variable_internal(variable, dimension, change_of_variables);
}

void Function::Implementation::add_variable_internal(double* variable,
                                                    int dimension,
                                                    std::shared_ptr<ChangeOfVariables> change_of_variables)
{
	this->local_storage_allocated = false;

	auto itr = variables.find(variable);
	if (itr != variables.end()) {
		AddedVariable& var_info = itr->second;

		if (var_info.user_dimension != dimension) {
			throw std::runtime_error("Function::add_variable: dimension mismatch "
			                         "with previously added variable.");
		}

		var_info.change_of_variables = change_of_variables;
		if (change_of_variables) {
			if (var_info.user_dimension != change_of_variables->x_dimension()) {
				throw std::runtime_error("Function::add_variable: "
			                             "x_dimension can not change.");
			}
			if (var_info.solver_dimension != change_of_variables->t_dimension()) {
				throw std::runtime_error("Function::add_variable: "
			                             "t_dimension can not change.");
			}
		}

		return;
	}

	AddedVariable& var_info = variables[variable];

	var_info.change_of_variables = change_of_variables;
	if (change_of_variables){
		if (dimension != change_of_variables->x_dimension()) {
			throw std::runtime_error("Function::add_variable: "
			                         "dimension does not match the change of variables.");
		}
		var_info.user_dimension   = change_of_variables->x_dimension();
		var_info.solver_dimension = change_of_variables->t_dimension();
	}
	else {
		var_info.user_dimension   = dimension;
		var_info.solver_dimension = dimension;
	}

	// Allocate local scratch spaces for evaluation.
	// We need as much space as the dimension of x.
	var_info.temp_space.resize(var_info.user_dimension);

	var_info.is_constant = false;

	// Give this variable a global index into a global
	// state vector.
	var_info.global_index = number_of_scalars;
	number_of_scalars += var_info.solver_dimension;
}

void Function::Implementation::set_constant(double* variable, bool is_constant)
{
	// Find the variable. This has to succeed.
	auto itr = variables.find(variable);
	if (itr == variables.end()) {
		throw std::runtime_error("Function::set_constant: variable not found.");
	}

	itr->second.is_constant = is_constant;

	// Recompute all global indices. Expensive!
	this->number_of_scalars = 0;
	for (auto& itr: variables) {
		auto& var_info = itr.second;

		if (! var_info.is_constant) {
			// Give this variable a global index into a global
			// state vector.
			var_info.global_index = this->number_of_scalars;
			this->number_of_scalars += var_info.solver_dimension;
		}
	}

	this->number_of_constants = 0;
	for (auto itr = variables.begin(); itr != variables.end(); ++itr) {
		auto& var_info = itr->second;
		
		if (var_info.is_constant) {
			// Give this variable a global index into a global
			// state vector.
			var_info.global_index = this->number_of_scalars + this->number_of_constants;
			this->number_of_constants += var_info.solver_dimension;
		}
	}

	this->local_storage_allocated = false;
}

void Function::set_constant(double* variable, bool is_constant)
{
	impl->set_constant(variable, is_constant);
}

void Function::add_term(std::shared_ptr<const Term> term, const std::vector<double*>& arguments)
{
	impl->local_storage_allocated = false;

	if (term->number_of_variables() != arguments.size()) {
		throw std::runtime_error("Function::add_term: incorrect number of arguments.");
	}

	// Check whether the variables exist.
	for (int var = 0; var < term->number_of_variables(); ++var) {
		auto var_itr = impl->variables.find(arguments[var]);
		if (var_itr == impl->variables.end()) {
			add_variable(arguments[var], term->variable_dimension(var));
		}
		// The x-dimension of the variable must match what is expected by the term.
		else if (var_itr->second.user_dimension != term->variable_dimension(var)) {
			throw std::runtime_error("Function::add_term: variable dimension does not match term.");
		}
	}

	//impl->added_terms.insert(term);

	impl->terms.push_back(AddedTerm());
	impl->terms.back().term = term;

	for (int var = 0; var < term->number_of_variables(); ++var) {
		// Look up this variable.
		AddedVariable& added_variable = impl->variables[arguments[var]];
		impl->terms.back().user_variables.push_back(&added_variable);
		// Stora a pointer to temporary storage for this variable.
		double* temp_space = &added_variable.temp_space[0];
		impl->terms.back().temp_variables.push_back(temp_space);
	}

	if (this->hessian_is_enabled) {
		// Create enough space for the hessian.
		impl->terms.back().hessian.resize(term->number_of_variables());
		for (int var0 = 0; var0 < term->number_of_variables(); ++var0) {
			impl->terms.back().hessian[var0].resize(term->number_of_variables());
			for (int var1 = 0; var1 < term->number_of_variables(); ++var1) {
				impl->terms.back().hessian[var0][var1].resize(term->variable_dimension(var0),
														term->variable_dimension(var1));
			}
		}
	}
}

size_t Function::get_number_of_terms() const
{
	return impl->terms.size();
}

void Function::set_number_of_threads(int num)
{
	#ifdef USE_OPENMP
		if (num <= 0) {
			throw std::runtime_error("Function::set_number_of_threads: "
			                         "invalid number of threads.");
		}
		impl->local_storage_allocated = false;
		impl->number_of_threads = num;
	#endif
}

void Function::Implementation::allocate_local_storage() const
{
	size_t max_arity = 1;
	int max_variable_dimension = 1;
	for (const auto& itr: variables) {
		max_variable_dimension = std::max(max_variable_dimension,
		                                  itr.second.user_dimension);
	}
	for (const auto& term: terms) {
		max_arity = std::max(max_arity, term.user_variables.size());
	}

	this->thread_gradient_scratch.resize(this->number_of_threads);
	this->thread_gradient_storage.resize(this->number_of_threads);
	for (int t = 0; t < this->number_of_threads; ++t) {
		this->thread_gradient_storage[t].resize(number_of_scalars + number_of_constants);
		this->thread_gradient_scratch[t].resize(max_arity);
		for (int var = 0; var < max_arity; ++var) {
			this->thread_gradient_scratch[t][var].resize(max_variable_dimension);
		}
	}

	this->local_storage_allocated = true;
}

void Function::add_term(std::shared_ptr<const Term> term, double* argument0)
{
	std::vector<double*> arguments;
	arguments.push_back(argument0);
	add_term(term, arguments);
}

void Function::add_term(std::shared_ptr<const Term> term, double* argument0, double* argument1)
{
	std::vector<double*> arguments;
	arguments.push_back(argument0);
	arguments.push_back(argument1);
	add_term(term, arguments);
}

void Function::print_timing_information(std::ostream& out) const
{
	out << "Function evaluations without gradient : " << evaluations_without_gradient << '\n';
	out << "Function evaluations with gradient    : " << evaluations_with_gradient << '\n';
	out << "Function evaluate time            : " << evaluate_time << '\n';
	out << "Function evaluate time (with g/H) : " << evaluate_with_hessian_time << '\n';
	out << "Function write g/H time           : " << write_gradient_hessian_time << '\n';
	out << "Function copy data time           : " << copy_time << '\n';
}

double Function::Implementation::evaluate_from_local_storage() const
{
	interface->evaluations_without_gradient++;
	double start_time = wall_time();

	double value = this->constant;
	// Go through and evaluate each term.
	// OpenMP requires a signed data type as the loop variable.
	#ifdef USE_OPENMP
		// Each thread needs to store a specific error.
		std::vector<std::exception_ptr> evaluation_errors(this->number_of_threads);

		#pragma omp parallel for reduction(+ : value) num_threads(this->number_of_threads)
	#endif
	// For loop has to be int for OpenMP.
	for (int i = 0; i < terms.size(); ++i) {
		#ifdef USE_OPENMP
			// The thread number calling this iteration.
			int t = omp_get_thread_num();
			// We need to catch all exceptions before leaving
			// the loop body.
			try {
		#endif

		// Evaluate the term .
		value += terms[i].term->evaluate(&terms[i].temp_variables[0]);

		#ifdef USE_OPENMP
			// We need to catch all exceptions before leaving
			// the loop body.
			}
			catch (...) {
				evaluation_errors[t] = std::current_exception();
			}
		#endif
	}

	#ifdef USE_OPENMP
		// Now that we are outside the OpenMP block, we can
		// rethrow exceptions.
		for (auto itr = evaluation_errors.begin(); itr != evaluation_errors.end(); ++itr) {
			// VS 2010 does not have conversion to bool or
			// operator !=.
			if ( !(*itr == std::exception_ptr())) {
				std::rethrow_exception(*itr);
			}
		}
	#endif

	interface->evaluate_time += wall_time() - start_time;
	return value;
}

double Function::evaluate(const Eigen::VectorXd& x) const
{
	// Copy values from the global vector x to the temporary storage
	// used for evaluating the term.
	impl->copy_global_to_local(x);

	return impl->evaluate_from_local_storage();
}

double Function::evaluate() const
{
	// Copy the user state to the local storage
	// for evaluation.
	impl->copy_user_to_local();

	return impl->evaluate_from_local_storage();
}

void Function::create_sparse_hessian(Eigen::SparseMatrix<double>* H) const
{
	std::vector<Eigen::Triplet<double> > indices;
	indices.reserve(impl->number_of_hessian_elements);
	impl->number_of_hessian_elements = 0;

	for (const auto& added_term: impl->terms) {
		auto& variables = added_term.user_variables;
		auto& term = added_term.term;

		// Put the hessian into the global hessian.
		for (int var0 = 0; var0 < term->number_of_variables(); ++var0) {
			if ( ! variables[var0]->is_constant) {

				size_t global_offset0 = variables[var0]->global_index;
				for (int var1 = 0; var1 < term->number_of_variables(); ++var1) {
					if ( ! variables[var1]->is_constant) {

						size_t global_offset1 = variables[var1]->global_index;
						for (size_t i = 0; i < term->variable_dimension(var0); ++i) {
							for (size_t j = 0; j < term->variable_dimension(var1); ++j) {
								int global_i = static_cast<int>(i + global_offset0);
								int global_j = static_cast<int>(j + global_offset1);
								indices.push_back(Eigen::Triplet<double>(global_i,
																		 global_j,
																		 1.0));
								impl->number_of_hessian_elements++;
							}
						}

					}
				}

			}
		}
	}

	auto n = static_cast<int>(impl->number_of_scalars);
	H->resize(n, n);
	H->setFromTriplets(indices.begin(), indices.end());
	H->makeCompressed();
}

void Function::Implementation::copy_global_to_local(const Eigen::VectorXd& x) const
{
	double start_time = wall_time();

	for (auto itr = variables.begin(); itr != variables.end(); ++itr) {

		if ( ! itr->second.is_constant) {
			if (itr->second.change_of_variables == nullptr) {
				for (int i = 0; i < itr->second.user_dimension; ++i) {
					itr->second.temp_space[i] = x[itr->second.global_index + i];
				}
			}
			else {
				itr->second.change_of_variables->t_to_x(
					&itr->second.temp_space[0],
					&x[itr->second.global_index]);
			}
		}
		else {
			// This variable is constant and is therefore not
			// present in the global vector x of variables.
			// Copy the constant from the user space instead.
			for (int i = 0; i < itr->second.user_dimension; ++i) {
				itr->second.temp_space[i] = itr->first[i];
			}
		}
	}

	interface->copy_time += wall_time() - start_time;
}

void Function::copy_user_to_global(Eigen::VectorXd* x) const
{
	impl->copy_user_to_global(x);
}

void Function::Implementation::copy_user_to_global(Eigen::VectorXd* x) const
{
	double start_time = wall_time();

	x->resize(this->number_of_scalars);
	for (const auto& itr: variables) {
		double* data = itr.first;
		const auto& variable = itr.second;

		if ( ! variable.is_constant) {
			if (variable.change_of_variables == nullptr) {
				for (int i = 0; i < variable.user_dimension; ++i) {
					(*x)[variable.global_index + i] = data[i];
				}
			}
			else {
				variable.change_of_variables->x_to_t(
					&(*x)[variable.global_index],
					data);
			}
		}
	}

	interface->copy_time += wall_time() - start_time;
}

void Function::copy_global_to_user(const Eigen::VectorXd& x) const
{
	impl->copy_global_to_user(x);
}

void Function::Implementation::copy_global_to_user(const Eigen::VectorXd& x) const
{
	double start_time = wall_time();

	for (const auto& itr: variables) {
		double* data = itr.first;
		const auto& variable = itr.second;

		if ( ! variable.is_constant) {
			if (variable.change_of_variables == nullptr) {
				for (int i = 0; i < variable.user_dimension; ++i) {
					data[i] = x[variable.global_index + i];
				}
			}
			else {
				variable.change_of_variables->t_to_x(
					data,
					&x[variable.global_index]);
			}
		}
	}

	interface->copy_time += wall_time() - start_time;
}

void Function::Implementation::copy_user_to_local() const
{
	double start_time = wall_time();

	for (const auto& itr: variables) {
		double* data = itr.first;
		const auto& variable = itr.second;

		// Both variables and constants are copied here.

		for (int i = 0; i < variable.user_dimension; ++i) {
			variable.temp_space[i] = data[i];
		}
	}

	interface->copy_time += wall_time() - start_time;
}

double Function::evaluate(const Eigen::VectorXd& x,
                          Eigen::VectorXd* gradient) const
{
	return this->evaluate(x, gradient, reinterpret_cast<Eigen::MatrixXd*>(0));
}


double Function::evaluate(const Eigen::VectorXd& x,
                          Eigen::VectorXd* gradient,
						  Eigen::MatrixXd* hessian) const
{
	return impl->evaluate(x, gradient, hessian);
}

double Function::Implementation::evaluate(const Eigen::VectorXd& x,
                                          Eigen::VectorXd* gradient,
						                  Eigen::MatrixXd* hessian) const
{
	interface->evaluations_with_gradient++;

	if (hessian && ! interface->hessian_is_enabled) {
		throw std::runtime_error("Function::evaluate: Hessian computation is not enabled.");
	}

	if (! this->local_storage_allocated) {
		this->allocate_local_storage();
	}

	// Copy values from the global vector x to the temporary storage
	// used for evaluating the term.
	this->copy_global_to_local(x);

	double start_time = wall_time();

	// Initialize each thread's global gradient.
	for (int t = 0; t < this->number_of_threads; ++t) {
		this->thread_gradient_storage[t].setZero();
	}

	double value = this->constant;

	// Go through and evaluate each term.
	// OpenMP requires a signed data type as the loop variable.
	#ifdef USE_OPENMP
		// Each thread needs to store a specific error.
		std::vector<std::exception_ptr> evaluation_errors(this->number_of_threads);

		#pragma omp parallel for reduction(+ : value) num_threads(this->number_of_threads)
	#endif
	for (int i = 0; i < terms.size(); ++i) {
		#ifdef USE_OPENMP
			// The thread number calling this iteration.
			int t = omp_get_thread_num();
			// We need to catch all exceptions before leaving
			// the loop body.
			try {
		#else
			int t = 0;
		#endif

		if (hessian) {
			// Evaluate the term and put its gradient and hessian
			// into local storage.
			value += terms[i].term->evaluate(&terms[i].temp_variables[0],
											 &this->thread_gradient_scratch[t],
											 &terms[i].hessian);
		}
		else {
			// Evaluate the term and put its gradient into local
			// storage.
			value += terms[i].term->evaluate(&terms[i].temp_variables[0],
											 &this->thread_gradient_scratch[t]);
		}

		// Put the gradient from the term into the thread's global gradient.
		const auto& variables = terms[i].user_variables;
		for (int var = 0; var < variables.size(); ++var) {

			if ( ! variables[var]->is_constant) {
				if (variables[var]->change_of_variables == nullptr) {
					// No change of variables, just copy the gradient.
					size_t global_offset = variables[var]->global_index;
					for (int i = 0; i < variables[var]->user_dimension; ++i) {
						this->thread_gradient_storage[t][global_offset + i] +=
							this->thread_gradient_scratch[t][var][i];
					}
				}
				else {
					// Transform the gradient from user space to solver space.
					size_t global_offset = variables[var]->global_index;
					if (global_offset < this->number_of_scalars) {
						variables[var]->change_of_variables->update_gradient(
							&this->thread_gradient_storage[t][global_offset],
							&x[global_offset],
							&this->thread_gradient_scratch[t][var][0]);
					}
				}
			}
		}

		#ifdef USE_OPENMP
			// We need to catch all exceptions before leaving
			// the loop body.
			}
			catch (...) {
				evaluation_errors[t] = std::current_exception();
			}
		#endif
	}

	#ifdef USE_OPENMP
		// Now that we are outside the OpenMP block, we can
		// rethrow exceptions.
		for (auto itr = evaluation_errors.begin(); itr != evaluation_errors.end(); ++itr) {
			// VS 2010 does not have conversion to bool or
			// operator !=.
			if ( !(*itr == std::exception_ptr())) {
				std::rethrow_exception(*itr);
			}
		}
	#endif

	interface->evaluate_with_hessian_time += wall_time() - start_time;
	start_time = wall_time();

	// Create the global gradient.
	if (gradient->size() != this->number_of_scalars) {
		gradient->resize(this->number_of_scalars);
	}
	gradient->setZero();
	// Sum the gradients from all different terms.
	for (int t = 0; t < this->number_of_threads; ++t) {
		(*gradient) += this->thread_gradient_storage[t].segment(0, this->number_of_scalars);
	}

	if (hessian) {
		// Create the global (dense) hessian.
		hessian->resize( static_cast<int>(this->number_of_scalars),
						 static_cast<int>(this->number_of_scalars));
		hessian->setZero();

		// Go through and evaluate each term.
		for (auto itr = terms.begin(); itr != terms.end(); ++itr) {
			auto& variables = itr->user_variables;

			// Put the hessian into the global hessian.
			for (int var0 = 0; var0 < itr->term->number_of_variables(); ++var0) {

				if ( ! variables[var0]->is_constant) {
					if (itr->user_variables[var0]->change_of_variables) {
						throw std::runtime_error("Change of variables not supported for Hessians");
					}

					size_t global_offset0 = variables[var0]->global_index;
					for (int var1 = 0; var1 < itr->term->number_of_variables(); ++var1) {
						size_t global_offset1 = variables[var1]->global_index;

						if ( ! variables[var1]->is_constant) {

							const Eigen::MatrixXd& part_hessian = itr->hessian[var0][var1];
							for (int i = 0; i < itr->term->variable_dimension(var0); ++i) {
								for (int j = 0; j < itr->term->variable_dimension(var1); ++j) {
									hessian->coeffRef(i + global_offset0, j + global_offset1) +=
										part_hessian(i, j);
								}
							}

						}
					}
				}

			}
		}
	}

	interface->write_gradient_hessian_time += wall_time() - start_time;
	return value;
}

double Function::evaluate(const Eigen::VectorXd& x,
                          Eigen::VectorXd* gradient,
						  Eigen::SparseMatrix<double>* hessian) const
{
	return impl->evaluate(x, gradient, hessian);
}

double Function::Implementation::evaluate(const Eigen::VectorXd& x,
                                          Eigen::VectorXd* gradient,
						                  Eigen::SparseMatrix<double>* hessian) const
{
	interface->evaluations_with_gradient++;

	if (! hessian) {
		throw std::runtime_error("Function::evaluate: hessian can not be null.");
	}

	if (! interface->hessian_is_enabled) {
		throw std::runtime_error("Function::evaluate: Hessian computation is not enabled.");
	}

	if (! this->local_storage_allocated) {
		this->allocate_local_storage();
	}

	// Copy values from the global vector x to the temporary storage
	// used for evaluating the term.
	this->copy_global_to_local(x);

	double start_time = wall_time();

	std::vector<Eigen::Triplet<double> > indices;
	indices.reserve(this->number_of_hessian_elements);
	this->number_of_hessian_elements = 0;

	interface->write_gradient_hessian_time += wall_time() - start_time;
	start_time = wall_time();

	// Initialize each thread's global gradient.
	for (int t = 0; t < this->number_of_threads; ++t) {
		this->thread_gradient_storage[t].setZero();
	}

	double value = this->constant;
	// Go through and evaluate each term.
	// OpenMP requires a signed data type as the loop variable.
	#ifdef USE_OPENMP
		// Each thread needs to store a specific error.
		std::vector<std::exception_ptr> evaluation_errors(this->number_of_threads);

		#pragma omp parallel for reduction(+ : value) num_threads(this->number_of_threads)
	#endif
	for (int i = 0; i < terms.size(); ++i) {
		#ifdef USE_OPENMP
			// The thread number calling this iteration.
			int t = omp_get_thread_num();
			// We need to catch all exceptions before leaving
			// the loop body.
			try {
		#else
			int t = 0;
		#endif

		// Evaluate the term and put its gradient and hessian
		// into local storage.
		value += terms[i].term->evaluate(&terms[i].temp_variables[0],
		                                 &this->thread_gradient_scratch[t],
		                                 &terms[i].hessian);

		// Put the gradient from the term into the thread's global gradient.
		const auto& variables = terms[i].user_variables;
		for (int var = 0; var < variables.size(); ++var) {

			if ( ! variables[var]->is_constant) {
				if (variables[var]->change_of_variables) {
					throw std::runtime_error("Change of variables not supported for sparse Hessian");
				}

				size_t global_offset = variables[var]->global_index;
				for (int i = 0; i < variables[var]->user_dimension; ++i) {
					this->thread_gradient_storage[t][global_offset + i] +=
						this->thread_gradient_scratch[t][var][i];
				}
			}
		}

		#ifdef USE_OPENMP
			// We need to catch all exceptions before leaving
			// the loop body.
			}
			catch (...) {
				evaluation_errors[t] = std::current_exception();
			}
		#endif
	}

	#ifdef USE_OPENMP
		// Now that we are outside the OpenMP block, we can
		// rethrow exceptions.
		for (auto itr = evaluation_errors.begin(); itr != evaluation_errors.end(); ++itr) {
			// VS 2010 does not have conversion to bool or
			// operator !=.
			if ( !(*itr == std::exception_ptr())) {
				std::rethrow_exception(*itr);
			}
		}
	#endif

	interface->evaluate_with_hessian_time += wall_time() - start_time;
	start_time = wall_time();

	// Create the global gradient.
	if (gradient->size() != this->number_of_scalars) {
		gradient->resize(this->number_of_scalars);
	}
	gradient->setZero();
	// Sum the gradients from all different terms.
	for (int t = 0; t < this->number_of_threads; ++t) {
		(*gradient) += this->thread_gradient_storage[t].segment(0, this->number_of_scalars);
	}

	// Collect the gradients and hessians from each term.
	for (auto itr = terms.begin(); itr != terms.end(); ++itr) {
		auto& variables = itr->user_variables;
		// Put the hessian into the global hessian.
		for (int var0 = 0; var0 < itr->term->number_of_variables(); ++var0) {
			if ( ! variables[var0]->is_constant) {

				size_t global_offset0 = variables[var0]->global_index;
				for (int var1 = 0; var1 < itr->term->number_of_variables(); ++var1) {
					if ( ! variables[var1]->is_constant) {

						size_t global_offset1 = variables[var1]->global_index;
						const Eigen::MatrixXd& part_hessian = itr->hessian[var0][var1];
						for (int i = 0; i < itr->term->variable_dimension(var0); ++i) {
							for (int j = 0; j < itr->term->variable_dimension(var1); ++j) {
								//std::cerr << "var=(" << var0 << ',' << var1 << ") ";
								//std::cerr << "ij=(" << i << ',' << j << ") ";
								//std::cerr << "writing to (" << i + global_offset0 << ',' << j + global_offset1 << ")\n";
								//hessian->coeffRef(i + global_offset0, j + global_offset1) +=
								//	part_hessian(i, j);
								int global_i = static_cast<int>(i + global_offset0);
								int global_j = static_cast<int>(j + global_offset1);
								indices.push_back(Eigen::Triplet<double>(global_i,
																		 global_j,
																		 part_hessian(i, j)));
								this->number_of_hessian_elements++;
							}
						}
					}

				}
			}

		}
	}

	hessian->setFromTriplets(indices.begin(), indices.end());
	//hessian->makeCompressed();

	interface->write_gradient_hessian_time += wall_time() - start_time;

	return value;
}

Interval<double> Function::evaluate(const std::vector<Interval<double>>& x) const
{
	return impl->evaluate(x);
}

Interval<double>  Function::Implementation::evaluate(const std::vector<Interval<double>>& x) const
{
	interface->evaluations_without_gradient++;
	double start_time = wall_time();

	std::vector<const Interval<double> *> scratch_space;

	Interval<double> value = this->constant;
	// Go through and evaluate each term.
	for (int i = 0; i < terms.size(); ++i) {
		// Evaluate the term.
		scratch_space.clear();
		for (auto itr = terms[i].user_variables.begin();
		     itr != terms[i].user_variables.end(); ++itr) {
			const auto& var = *itr;
			scratch_space.push_back(&x[var->global_index]);
		}
		value += terms[i].term->evaluate_interval(&scratch_space[0]);
	}

	interface->evaluate_time += wall_time() - start_time;
	return value;
}

void Function::write_to_stream(std::ostream& out) const
{
	using namespace std;

	// Use high precision.
	out << setprecision(30);

	// Write version to stream;
	out << "spii::function" << endl;
	out << 1 << endl;
	// Write the representation of a reasonably complicated class to
	// the file. We can then check that the compiler-dependent format
	// matches.
	out << TermFactory::fix_name(typeid(std::vector<std::map<double,int>>).name()) << endl;

	out << impl->terms.size() << endl;
	out << impl->variables.size() << endl;
	out << impl->number_of_scalars << endl;
	out << impl->constant << endl;

	vector<pair<std::size_t, std::size_t>> variable_dimensions; 
	for (const auto& variable : impl->variables) {
		if (variable.second.change_of_variables != nullptr) {
			throw runtime_error("Function::write_to_stream: Change of variables not allowed.");
		}

		variable_dimensions.emplace_back(variable.second.global_index, variable.second.user_dimension);
	}
	sort(variable_dimensions.begin(), variable_dimensions.end());
	for (const auto& variable : variable_dimensions) {
		out << variable.second << " ";
	}
	out << endl;

	Eigen::VectorXd x;
	this->copy_user_to_global(&x);
	for (int i = 0; i < impl->number_of_scalars; ++i) {
		out << x[i] << " ";
	}
	out << endl;

	for (const auto& added_term : impl->terms) {
		string term_name = TermFactory::fix_name(typeid(*added_term.term).name());
		out << term_name << endl;
		out << added_term.user_variables.size() << endl;
		for (const auto& var : added_term.user_variables) {
			out << var->global_index << " ";
		}
		out << endl;
		out << *added_term.term << endl;
	}
}

void Function::read_from_stream(std::istream& in, std::vector<double>* user_space, const TermFactory& factory)
{
	using namespace std;

	impl->clear();

	auto check = [&in](const char* variable_name) 
	{ 
		if (!in) {
			std::string msg = "Function::read_from_stream: Reading ";
			msg += variable_name;
			msg += " failed.";
			throw runtime_error(msg.c_str()); 
		}
	};
	#define read_and_check(var) in >> var; check(#var); //cout << #var << " = " << var << endl;

 	// TODO: Clear f.

	string spii_function;
	read_and_check(spii_function);
	if (spii_function != "spii::function") {
		throw runtime_error("Function::read_from_stream: Not a function stream.");
	}
	int version;
	read_and_check(version);
	string compiler_type_format;
	read_and_check(compiler_type_format);
	if (compiler_type_format
	    != TermFactory::fix_name(typeid(std::vector<std::map<double,int>>).name()))
	{
		throw runtime_error("Function::read_from_stream: Type format does not match. "
		                    "Files can not be shared between compilers.");
	}

	unsigned number_of_terms;
	read_and_check(number_of_terms);
	unsigned number_of_variables;
	read_and_check(number_of_variables);
	unsigned number_of_scalars;
	read_and_check(number_of_scalars);
	read_and_check(impl->constant);

	user_space->resize(number_of_scalars);
	int current_var = 0;
	for (unsigned i = 0; i < number_of_variables; ++i) {
		int variable_dimension;
		read_and_check(variable_dimension);
		this->add_variable(&user_space->at(current_var), variable_dimension);
		current_var += variable_dimension;
	}
	if (current_var != number_of_scalars) {
		throw runtime_error("Function::read_from_stream: Not enough variables in stream.");
	}

	for (unsigned i = 0; i < number_of_scalars; ++i) {
		read_and_check(user_space->at(i));
	}

	for (unsigned i = 0; i < number_of_terms; ++i) {
		std::string term_name;
		read_and_check(term_name);
		unsigned term_vars;
		read_and_check(term_vars);

		std::vector<double*> arguments;
		for (unsigned i = 0; i < term_vars; ++i) {
			int offset;
			read_and_check(offset);
			arguments.push_back(&user_space->at(offset));
		}

		auto term = std::shared_ptr<const Term>(factory.create(term_name, in));
		this->add_term(term, arguments);
	}

	#undef read_and_check
}

}  // namespace spii