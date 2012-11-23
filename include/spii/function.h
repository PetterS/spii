#ifndef SPII_FUNCTION_H
#define SPII_FUNCTION_H

#include <cstddef>
#include <map>
#include <set>
using std::size_t;

#include <Eigen/SparseCore>

#include <spii/term.h>

struct AddedVariable
{
	int dimension;
	size_t global_index;
	std::vector<double>  temp_space;
};

struct AddedTerm
{
	const Term* term;
	std::vector<double*> user_variables;
	// Temporary storage for a point, gradient and hessian.
	std::vector<double*> temp_variables;
	std::vector<Eigen::VectorXd> gradient;
	std::vector< std::vector<Eigen::MatrixXd> > hessian;
};

class Function
{
friend class Solver;
public:
	// Specifies whether the function should delete the terms
	// added to it. Default is for the function to delete them.
	// Note that it is still safe to add the same term multiple
	// times.
	enum {DeleteTerms, DoNotDeleteTerms} term_deletion; 

	Function();
	~Function();

	void add_variable(double* variable, int dimension);

	size_t get_number_of_variables() const
	{
		return variables.size();
	}

	size_t get_number_of_scalars() const
	{
		return number_of_scalars;
	}

	void add_term(const Term* term, const std::vector<double*>& arguments);
	void add_term(const Term* term, double* argument1);
	void add_term(const Term* term, double* argument1, double* argument2);

	size_t get_number_of_terms() const
	{
		return terms.size();
	}

	// Evaluation using the data in the user-provided space.
	double evaluate() const;

	// Evaluation using a global vector.
	double evaluate(const Eigen::VectorXd& x) const;

	template<typename Matrix>
	double evaluate(const Eigen::VectorXd& x, 
	            Eigen::VectorXd* gradient,
	            Matrix* hessian) const;

	// Create a sparse matrix with the correct sparsity pattern.
	void create_sparse_hessian(Eigen::SparseMatrix<double>* H) const;

protected:
	size_t global_index(double* variable) const;
	void copy_global_to_local(const Eigen::VectorXd& x) const;
	void copy_global_to_user(const Eigen::VectorXd& x) const;
	void copy_user_to_global(Eigen::VectorXd* x) const;

	// A set of all terms added to the function. This is
	// used when the function is destructed.
	std::set<const Term*> added_terms;

	// Has to be mutable because the temporary storage
	// needs to be written to.
	mutable std::map<double*, AddedVariable> variables;
	size_t number_of_scalars;

	// Has to be mutable because the temporary storage
	// needs to be written to.
	mutable std::vector<AddedTerm> terms;
};


template<typename Matrix>
double Function::evaluate(const Eigen::VectorXd& x,
                          Eigen::VectorXd* gradient,
						  Matrix* hessian) const
{
	// Copy values from the global vector x to the temporary storage
	// used for evaluating the term.
	this->copy_global_to_local(x);
	double value = 0;
	// Create the global gradient.
	gradient->resize(this->number_of_scalars);
	gradient->setConstant(0.0);
	// Create the global (dense) hessian.
	hessian->resize(this->number_of_scalars, this->number_of_scalars);
	//hessian->setConstant(0.0);
	(*hessian) *= 0.0;
	// Go through and evaluate each term.
	for (auto itr = terms.begin(); itr != terms.end(); ++itr) {
		// Evaluate the term and put its gradient and hessian
		// into local storage.
		value += itr->term->evaluate(&itr->temp_variables[0], 
		                             &itr->gradient,
		                             &itr->hessian);
		// Put the gradient into the global gradient.
		for (int var = 0; var < itr->term->number_of_variables(); ++var) {
			size_t global_offset = this->global_index(itr->user_variables[var]);
			for (int i = 0; i < itr->term->variable_dimension(var); ++i) {
				(*gradient)[global_offset + i] += itr->gradient[var][i];
			}
		}
		// Put the hessian into the global hessian.
		for (int var0 = 0; var0 < itr->term->number_of_variables(); ++var0) {
			size_t global_offset0 = this->global_index(itr->user_variables[var0]);
			for (int var1 = 0; var1 < itr->term->number_of_variables(); ++var1) {
				size_t global_offset1 = this->global_index(itr->user_variables[var1]);
				Eigen::MatrixXd& part_hessian = itr->hessian[var0][var1];
				for (int i = 0; i < itr->term->variable_dimension(var0); ++i) {
					for (int j = 0; j < itr->term->variable_dimension(var1); ++j) {
						//std::cerr << "var=(" << var0 << ',' << var1 << ") ";
						//std::cerr << "ij=(" << i << ',' << j << ") ";
						//std::cerr << "writing to (" << i + global_offset0 << ',' << j + global_offset1 << ")\n";
						hessian->coeffRef(i + global_offset0, j + global_offset1) +=
							part_hessian(i, j);
					}
				}
			}
		}
	}
	return value;
}

#endif SPII_FUNCTION_H
