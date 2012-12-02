
#include <iostream>
#include <stdexcept>

#include <spii/function.h>
#include <spii/spii.h>

namespace spii {

Function::Function()
{
	this->number_of_scalars = 0;
	this->term_deletion = DeleteTerms;

	this->hessian_is_enabled = true;

	this->number_of_hessian_elements = 0;

	this->evaluations_without_gradient = 0;
	this->evaluations_with_gradient    = 0;

	this->evaluate_time               = 0;
	this->evaluate_with_hessian_time  = 0;
	this->write_gradient_hessian_time = 0;
	this->copy_time                   = 0;

	this->number_of_threads = omp_get_max_threads();

	this->finalize_called = false;
}

Function::~Function()
{
	if (this->term_deletion == DeleteTerms) {
		for (auto itr = added_terms.begin(); itr != added_terms.end(); ++itr) {
			delete *itr;
		}
	}
}

void Function::add_variable(double* variable, int dimension)
{
	this->finalize_called = false;

	auto itr = variables.find(variable);
	if (itr != variables.end()) {
		if (itr->second.dimension != dimension) {
			throw std::runtime_error("Function::add_variable: dimension mismatch.");
		}
		return;
	}
	AddedVariable& var_info = variables[variable];
	var_info.dimension = dimension;
	var_info.global_index = number_of_scalars;
	var_info.temp_space.resize(dimension);
	number_of_scalars += dimension;
}


void Function::add_term(const Term* term, const std::vector<double*>& arguments)
{
	this->finalize_called = false;

	if (term->number_of_variables() != arguments.size()) {
		throw std::runtime_error("Function::add_term: incorrect number of arguments.");
	}
	for (int var = 0; var < term->number_of_variables(); ++var) {
		auto var_itr = variables.find(arguments[var]);
		if (var_itr == variables.end()) {
			throw std::runtime_error("Function::add_term: unknown variable.");
		}
		if (var_itr->second.dimension != term->variable_dimension(var)) {
			throw std::runtime_error("Function::add_term: variable dimension does not match term.");
		}
	}
	
	added_terms.insert(term);

	terms.push_back(AddedTerm());
	terms.back().term = term;

	for (int var = 0; var < term->number_of_variables(); ++var) {
		// Look up this variable.
		AddedVariable& added_variable = this->variables[arguments[var]];
		terms.back().user_variables.push_back(&added_variable);
		// Stora a pointer to temporary storage for this variable.
		double* temp_space = &added_variable.temp_space[0];
		terms.back().temp_variables.push_back(temp_space);
	}

	if (this->hessian_is_enabled) {
		// Create enough space for the hessian.
		terms.back().hessian.resize(term->number_of_variables());
		for (int var0 = 0; var0 < term->number_of_variables(); ++var0) {
			terms.back().hessian[var0].resize(term->number_of_variables());
			for (int var1 = 0; var1 < term->number_of_variables(); ++var1) {
				terms.back().hessian[var0][var1].resize(term->variable_dimension(var0),
														term->variable_dimension(var1));
			}
		}
	}
}

void Function::set_number_of_threads(int num)
{
	this->finalize_called = false;
	this->number_of_threads = num;
}

void Function::finalize() const
{
	size_t max_arity = 1;
	int max_variable_dimension = 1;
	for (auto itr = variables.begin(); itr != variables.end(); ++itr) {
		max_variable_dimension = std::max(max_variable_dimension,
		                                  itr->second.dimension);
	}
	for (auto itr = terms.begin(); itr != terms.end(); ++itr) {
		max_arity = std::max(max_arity, itr->user_variables.size());
	}

	this->thread_gradient_scratch.resize(this->number_of_threads);
	this->thread_gradient_storage.resize(this->number_of_threads);
	for (int t = 0; t < this->number_of_threads; ++t) {
		this->thread_gradient_storage[t].resize(number_of_scalars);
		this->thread_gradient_scratch[t].resize(max_arity);
		for (int var = 0; var < max_arity; ++var) {
			this->thread_gradient_scratch[t][var].resize(max_variable_dimension);
		}
	}

	this->finalize_called = true;
}

void Function::add_term(const Term* term, double* argument0)
{
	std::vector<double*> arguments;
	arguments.push_back(argument0);
	add_term(term, arguments);
}

void Function::add_term(const Term* term, double* argument0, double* argument1)
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

double Function::evaluate(const Eigen::VectorXd& x) const
{
	this->evaluations_without_gradient++;

	// Copy values from the global vector x to the temporary storage
	// used for evaluating the term.
	this->copy_global_to_local(x);

	double start_time = wall_time();

	double value = 0;
	// Go through and evaluate each term.
	// OpenMP requires a signed data type as the loop variable.
	#ifdef USE_OPENMP
	#pragma omp parallel for reduction(+ : value) num_threads(this->number_of_threads)
	#endif
	for (int i = 0; i < terms.size(); ++i) {
		// Evaluate the term .
		value += terms[i].term->evaluate(&terms[i].temp_variables[0]);
	}

	this->evaluate_time += wall_time() - start_time;
	return value;
}

double Function::evaluate() const
{
	// This overload copies a lot of data. First from user space
	// to a global vector, then from the global vector to temporary
	// storage.
	Eigen::VectorXd x;
	this->copy_user_to_global(&x);

	return evaluate(x);
}

void Function::create_sparse_hessian(Eigen::SparseMatrix<double>* H) const
{
	std::vector<Eigen::Triplet<double> > indices;
	indices.reserve(this->number_of_hessian_elements);
	this->number_of_hessian_elements = 0;

	for (auto itr = terms.begin(); itr != terms.end(); ++itr) {
		// Put the hessian into the global hessian.
		for (int var0 = 0; var0 < itr->term->number_of_variables(); ++var0) {
			size_t global_offset0 = itr->user_variables[var0]->global_index;
			for (int var1 = 0; var1 < itr->term->number_of_variables(); ++var1) {
				size_t global_offset1 = itr->user_variables[var1]->global_index;
				for (size_t i = 0; i < itr->term->variable_dimension(var0); ++i) {
					for (size_t j = 0; j < itr->term->variable_dimension(var1); ++j) {
						int global_i = static_cast<int>(i + global_offset0);
						int global_j = static_cast<int>(j + global_offset1);
						indices.push_back(Eigen::Triplet<double>(global_i,
						                                         global_j,
						                                         1.0));
						this->number_of_hessian_elements++;
					}
				}
			}
		}
	}
	H->resize(static_cast<int>(this->number_of_scalars),
	          static_cast<int>(this->number_of_scalars));
	H->setFromTriplets(indices.begin(), indices.end());
	H->makeCompressed();
}

void Function::copy_global_to_local(const Eigen::VectorXd& x) const
{
	double start_time = wall_time();

	for (auto itr = variables.begin(); itr != variables.end(); ++itr) {
		for (int i = 0; i < itr->second.dimension; ++i) {
			itr->second.temp_space[i] = x[itr->second.global_index + i];
		}
	}

	this->copy_time += wall_time() - start_time;
}

void Function::copy_user_to_global(Eigen::VectorXd* x) const
{
	double start_time = wall_time();

	x->resize(this->number_of_scalars);
	for (auto itr = variables.begin(); itr != variables.end(); ++itr) {
		for (int i = 0; i < itr->second.dimension; ++i) {
			(*x)[itr->second.global_index + i] = itr->first[i];
		}
	}

	this->copy_time += wall_time() - start_time;
}

void Function::copy_global_to_user(const Eigen::VectorXd& x) const
{
	double start_time = wall_time();

	for (auto itr = variables.begin(); itr != variables.end(); ++itr) {
		for (int i = 0; i < itr->second.dimension; ++i) {
			itr->first[i] = x[itr->second.global_index + i];
		}
	}

	this->copy_time += wall_time() - start_time;
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
	this->evaluations_with_gradient++;

	if (hessian && ! this->hessian_is_enabled) {
		throw std::runtime_error("Function::evaluate: Hessian computation is not enabled.");
	}

	if (! this->finalize_called) {
		this->finalize();
	}

	// Copy values from the global vector x to the temporary storage
	// used for evaluating the term.
	this->copy_global_to_local(x);

	double start_time = wall_time();

	// Initialize each thread's global gradient.
	for (int t = 0; t < this->number_of_threads; ++t) {
		this->thread_gradient_storage[t].setZero();
	}

	double value = 0;

	// Go through and evaluate each term.
	// OpenMP requires a signed data type as the loop variable.
	#ifdef USE_OPENMP
	#pragma omp parallel for reduction(+ : value) num_threads(this->number_of_threads)
	#endif
	for (int i = 0; i < terms.size(); ++i) {
		// The thread number calling this iteration.
		int t = omp_get_thread_num();

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
			size_t global_offset = variables[var]->global_index;
			for (int i = 0; i < variables[var]->dimension; ++i) {
				this->thread_gradient_storage[t][global_offset + i] +=
					this->thread_gradient_scratch[t][var][i];
			}
		}
	}
	
	this->evaluate_with_hessian_time += wall_time() - start_time;
	start_time = wall_time();

	// Create the global gradient.
	if (gradient->size() != this->number_of_scalars) {
		gradient->resize(this->number_of_scalars);
	}
	gradient->setZero();
	// Sum the gradients from all different terms.
	for (int t = 0; t < this->number_of_threads; ++t) {
		(*gradient) += this->thread_gradient_storage[t];
	}

	if (hessian) {
		// Create the global (dense) hessian.
		hessian->resize( static_cast<int>(this->number_of_scalars),
						 static_cast<int>(this->number_of_scalars));
		hessian->setZero();

		// Go through and evaluate each term.
		for (auto itr = terms.begin(); itr != terms.end(); ++itr) {
			// Put the hessian into the global hessian.
			for (int var0 = 0; var0 < itr->term->number_of_variables(); ++var0) {
				size_t global_offset0 = itr->user_variables[var0]->global_index;
				for (int var1 = 0; var1 < itr->term->number_of_variables(); ++var1) {
					size_t global_offset1 = itr->user_variables[var1]->global_index;
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

	this->write_gradient_hessian_time += wall_time() - start_time;
	return value;
}

double Function::evaluate(const Eigen::VectorXd& x,
                          Eigen::VectorXd* gradient,
						  Eigen::SparseMatrix<double>* hessian) const
{
	this->evaluations_with_gradient++;

	if (! this->hessian_is_enabled) {
		throw std::runtime_error("Function::evaluate: Hessian computation is not enabled.");
	}

	if (! this->finalize_called) {
		this->finalize();
	}

	// Copy values from the global vector x to the temporary storage
	// used for evaluating the term.
	this->copy_global_to_local(x);

	double start_time = wall_time();
	
	std::vector<Eigen::Triplet<double> > indices;
	indices.reserve(this->number_of_hessian_elements);
	this->number_of_hessian_elements = 0;

	this->write_gradient_hessian_time += wall_time() - start_time;
	start_time = wall_time();

	// Initialize each thread's global gradient.
	for (int t = 0; t < this->number_of_threads; ++t) {
		this->thread_gradient_storage[t].setZero();
	}

	double value = 0;
	// Go through and evaluate each term.
	// OpenMP requires a signed data type as the loop variable.
	#ifdef USE_OPENMP
	#pragma omp parallel for reduction(+ : value) num_threads(this->number_of_threads)
	#endif
	for (int i = 0; i < terms.size(); ++i) {
		// The thread number calling this iteration.
		int t = omp_get_thread_num();

		// Evaluate the term and put its gradient and hessian
		// into local storage.
		value += terms[i].term->evaluate(&terms[i].temp_variables[0], 
		                                 &this->thread_gradient_scratch[t],
		                                 &terms[i].hessian);

		// Put the gradient from the term into the thread's global gradient.
		const auto& variables = terms[i].user_variables;
		for (int var = 0; var < variables.size(); ++var) {
			size_t global_offset = variables[var]->global_index;
			for (int i = 0; i < variables[var]->dimension; ++i) {
				this->thread_gradient_storage[t][global_offset + i] +=
					this->thread_gradient_scratch[t][var][i];
			}
		}
	}
	
	this->evaluate_with_hessian_time += wall_time() - start_time;
	start_time = wall_time();

	// Create the global gradient.
	if (gradient->size() != this->number_of_scalars) {
		gradient->resize(this->number_of_scalars);
	}
	gradient->setZero();
	// Sum the gradients from all different terms.
	for (int t = 0; t < this->number_of_threads; ++t) {
		(*gradient) += this->thread_gradient_storage[t];
	}

	// Collect the gradients and hessians from each term.
	for (auto itr = terms.begin(); itr != terms.end(); ++itr) {
		// Put the hessian into the global hessian.
		for (int var0 = 0; var0 < itr->term->number_of_variables(); ++var0) {
			size_t global_offset0 = itr->user_variables[var0]->global_index;
			for (int var1 = 0; var1 < itr->term->number_of_variables(); ++var1) {
				size_t global_offset1 = itr->user_variables[var1]->global_index;
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

	hessian->setFromTriplets(indices.begin(), indices.end());
	//hessian->makeCompressed();

	this->write_gradient_hessian_time += wall_time() - start_time;

	return value;
}

}  // namespace spii