// Petter Strandmark 2012.
#ifndef SPII_FUNCTION_H
#define SPII_FUNCTION_H
// This header defines the Function class which is used
// to store an objective function to be optimized.

#include <cstddef>
#include <map>
#include <memory>
#include <set>
using std::size_t;

#include <Eigen/SparseCore>

#include <spii/spii.h>
#include <spii/auto_diff_change_of_variables.h>
#include <spii/change_of_variables.h>
#include <spii/interval.h>
#include <spii/term.h>

namespace spii {

// Note on change of variables.
// The Function supports a change of variables, where the solver
// will see one set of variables and the evaluation function
// another. The variable change is specified with a ChangeOfVariables
// object. Each variable has a
//
//  * user_dimension   -- the dimension the Term object sees for
//                        evaluation.
//  * solver_dimension -- the dimension of the variables the solver
//                        sees.
//
//  If no ChangeOfVariables is used, these will be equal and the
//  solvers and terms will see identical values.
//

class SPII_API Function
{
friend class Solver;
public:
	// Specifies whether the function should delete the terms
	// added to it. Default is for the function to delete them.
	// Note that it is still safe to add the same term multiple
	// times.
	enum {DeleteTerms, DoNotDeleteTerms} term_deletion;

	// Specifies whether the function should be prepared to compute
	// the Hessian matrix, which is is not needed for L-BFGS. This
	// setting only affects the amount of temporary space allocated.
	// Default: true.
	bool hessian_is_enabled;

	Function();
	~Function();

	// Adds a variable to the function. All variables must be added
	// before any terms containing them are added.
	void add_variable(double* variable, int dimension);

	// Adds a variable to the function, with a change of variables.
	// Takes ownership of change and will delete it when the function
	// is destroyed.
	template<typename Change>
	void add_variable(double* variable,
	                  int dimension,
					  Change* change)
	{
		add_variable_internal(variable, dimension,
			 new AutoDiffChangeOfVariables<Change>(change));
	}

	// Returns the current number of variables the function contains.
	size_t get_number_of_variables() const;

	// Returns the current number of scalars the function contains.
	// (each variable contains of one or several scalars.)
	size_t get_number_of_scalars() const;

	// Sets the number of threads the Function should use when evaluating.
	// Default: number of cores available.
	void set_number_of_threads(int num);

	// Adds a new term to the function. Will throw an error if a variable
	// is not already added to the function, or if it does not match the
	// dimensionality required by the Term.
	//
	// The term_deletion member specified whether the Function is responsible
	// for calling delete on the term. In any case, it is safe to add the
	// same term twice.
	void add_term(const Term* term, const std::vector<double*>& arguments);
	void add_term(const Term* term, double* argument1);
	void add_term(const Term* term, double* argument1, double* argument2);

	// Returns the current number of terms contained in the function.
	size_t get_number_of_terms() const;

	// Evaluation using the data in the user-provided space.
	double evaluate() const;

	// Evaluation using a global vector.
	double evaluate(const Eigen::VectorXd& x) const;

	// Evaluate the function and compute the gradient at the point x.
	double evaluate(const Eigen::VectorXd& x,
	                Eigen::VectorXd* gradient) const;

	// Evaluate the function and compute the gradient and Hessian matrix
	// at the point x. Dense version.
	double evaluate(const Eigen::VectorXd& x,
	                Eigen::VectorXd* gradient,
	                Eigen::MatrixXd* hessian) const;

	// Same functionality as above, but for a sparse Hessian.
	double evaluate(const Eigen::VectorXd& x,
	                Eigen::VectorXd* gradient,
	                Eigen::SparseMatrix<double>* hessian) const;

	Interval<double> evaluate(const std::vector<Interval<double>>& x) const;

	// Copies variables from a global vector x to the storage
	// provided by the user.
	void copy_global_to_user(const Eigen::VectorXd& x) const;

	// Copies variables from a the storage provided by the user
	// to a global vector x.
	void copy_user_to_global(Eigen::VectorXd* x) const;

	// Create a sparse matrix with the correct sparsity pattern.
	void create_sparse_hessian(Eigen::SparseMatrix<double>* H) const;

	// Used to record the time of some operations. Each time an operation
	// is performed, the time taken is added to the appropiate variable.
	mutable int evaluations_without_gradient;
	mutable int evaluations_with_gradient;
	mutable double evaluate_time;
	mutable double evaluate_with_hessian_time;
	mutable double write_gradient_hessian_time;
	mutable double copy_time;

	// Prints the recorded timing information.
	void print_timing_information(std::ostream& out) const;

private:

	// Present here because it is called by a templated function above.
	void add_variable_internal(double* variable,
	                           int dimension,
	                           ChangeOfVariables* change_of_variables = 0);

	// Disallow copying for now.
	Function(const Function&);

	class Implementation;
	// unique_pointer would have been nice, but there are issues
	// with sharing these objects across DLL boundaries in VC++.
	Implementation* impl;
};

}  // namespace spii

#endif
