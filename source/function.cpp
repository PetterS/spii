
#include <iostream>
#include <stdexcept>

#include <spii/function.h>

Function::Function()
{
	this->number_of_scalars = 0;
	this->term_deletion = DeleteTerms;
}

Function::~Function()
{
	for (auto itr = added_terms.begin(); itr != added_terms.end(); ++itr) {
		delete *itr;
	}
}

void Function::add_variable(double* variable, int dimension)
{
	auto itr = variables.find(variable);
	if (itr != variables.end()) {
		if (itr->second.dimension != dimension) {
			throw std::runtime_error("Function::add_variable: dimension mismatch.");
		}
	}
	AddedVariable& var_info = variables[variable];
	var_info.dimension = dimension;
	var_info.global_index = number_of_scalars;
	var_info.temp_space.resize(dimension);
	number_of_scalars += dimension;
}


void Function::add_term(const Term* term, const std::vector<double*>& arguments)
{
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
	terms.back().user_variables = arguments;

	// Create enough space for the temporary point.
	int storage_needed = 0;
	for (int var = 0; var < term->number_of_variables(); ++var) {
		storage_needed += term->variable_dimension(var);
	}

	for (int var = 0; var < term->number_of_variables(); ++var) {
		// Stora a pointer to temporary storage for this variable.
		double* temp_space = &this->variables[arguments[var]].temp_space[0];
		terms.back().temp_variables.push_back(temp_space);
		// Create space for the gradient of this variable.
		terms.back().gradient.push_back(Eigen::VectorXd(term->variable_dimension(var)));
	}

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

double Function::evaluate(const Eigen::VectorXd& x) const
{
	// Copy values from the global vector x to the temporary storage
	// used for evaluating the term.
	this->copy_global_to_local(x);
	double value = 0;
	for (auto itr = terms.begin(); itr != terms.end(); ++itr) {
		// Evaluate this term.
		value += itr->term->evaluate(&itr->temp_variables[0]);
	}
	return value;
}

double Function::evaluate() const
{
	double value = 0;
	for (auto itr = terms.begin(); itr != terms.end(); ++itr) {
		value += itr->term->evaluate(&itr->user_variables[0]);
	}
	return value;
}

void Function::create_sparse_hessian(Eigen::SparseMatrix<double>* H) const
{
	std::vector<Eigen::Triplet<double> > indices;
	for (auto itr = terms.begin(); itr != terms.end(); ++itr) {
		// Put the hessian into the global hessian.
		for (int var0 = 0; var0 < itr->term->number_of_variables(); ++var0) {
			size_t global_offset0 = this->global_index(itr->user_variables[var0]);
			for (int var1 = 0; var1 < itr->term->number_of_variables(); ++var1) {
				size_t global_offset1 = this->global_index(itr->user_variables[var1]);
				for (size_t i = 0; i < itr->term->variable_dimension(var0); ++i) {
					for (size_t j = 0; j < itr->term->variable_dimension(var1); ++j) {
						indices.push_back(Eigen::Triplet<double>(i + global_offset0,
						                                         j + global_offset1,
						                                         1.0));
					}
				}
			}
		}
	}
	H->resize(this->number_of_scalars, this->number_of_scalars);
	H->setFromTriplets(indices.begin(), indices.end());
	H->makeCompressed();
}

size_t Function::global_index(double* variable) const
{
	auto itr = variables.find(variable);
	if (itr == variables.end()) {
		throw std::runtime_error("Function::global_index: Could not find variable");
	}
	return itr->second.global_index;
}


void Function::copy_global_to_local(const Eigen::VectorXd& x) const
{
	for (auto itr = variables.begin(); itr != variables.end(); ++itr) {
		for (int i = 0; i < itr->second.dimension; ++i) {
			itr->second.temp_space[i] = x[itr->second.global_index + i];
		}
	}
}

void Function::copy_user_to_global(Eigen::VectorXd* x) const
{
	x->resize(this->number_of_scalars);
	for (auto itr = variables.begin(); itr != variables.end(); ++itr) {
		for (int i = 0; i < itr->second.dimension; ++i) {
			(*x)[itr->second.global_index + i] = itr->first[i];
		}
	}
}

void Function::copy_global_to_user(const Eigen::VectorXd& x) const
{
	for (auto itr = variables.begin(); itr != variables.end(); ++itr) {
		for (int i = 0; i < itr->second.dimension; ++i) {
			itr->first[i] = x[itr->second.global_index + i];
		}
	}
}