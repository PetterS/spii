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

	double evaluate(const Eigen::VectorXd& x, 
	            Eigen::VectorXd* gradient,
	            Eigen::MatrixXd* hessian) const;

	double evaluate(const Eigen::VectorXd& x, 
	            Eigen::VectorXd* gradient,
	            Eigen::SparseMatrix<double>* hessian) const;

	// Create a sparse matrix with the correct sparsity pattern.
	void create_sparse_hessian(Eigen::SparseMatrix<double>* H) const;

	// Used to record the time of some operations. Each time an operation
	// is performed, the time taken is added to the appropiate variable.
	mutable double evaluate_time;
	mutable double evaluate_with_hessian_time;
	mutable double write_gradient_hessian_time;
	mutable double copy_time;

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

	// Stored how many element were used the last time the Hessian
	// was created.
	mutable size_t number_of_hessian_elements;
};


#endif
