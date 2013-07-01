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
#include <spii/term_factory.h>

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
	// Specifies whether the function should be prepared to compute
	// the Hessian matrix, which is is not needed for L-BFGS. This
	// setting only affects the amount of temporary space allocated.
	// Default: true.
	bool hessian_is_enabled;

	Function();
	~Function();
	// Copying may be expensive for large functions.
	Function(const Function&);
	Function& operator = (const Function&);

	// Adds a function to another. Neither function can have any change
	// of variables defined (ambiguous).
	Function& operator += (const Function&);
	Function& operator += (double constant_value);

	// Adds a new term to the function. Will throw an error if a variable
	// is already added to the function, and it does not match the
	// dimensionality required by the Term.
	//
	// If the variable has not previously been used, it will be added.
	//
	// The term_deletion member specified whether the Function is responsible
	// for calling delete on the term. In any case, it is safe to add the
	// same term twice.
	void add_term(std::shared_ptr<const Term> term, const std::vector<double*>& arguments);
	void add_term(std::shared_ptr<const Term> term, double* argument1);
	void add_term(std::shared_ptr<const Term> term, double* argument1, double* argument2);

	template<typename MyTerm>
	void add_term(double* argument1)
	{
		add_term(std::make_shared<MyTerm>(), argument1);
	}

	template<typename MyTerm>
	void add_term(double* argument1, double* argument2)
	{
		add_term(std::make_shared<MyTerm>(), argument1, argument2);
	}

	template<typename MyTerm>
	void add_term(double* argument1, double* argument2, double* argument3)
	{
		add_term(std::make_shared<MyTerm>(), argument1, argument2, argument3);
	}

	// Returns the current number of terms contained in the function.
	size_t get_number_of_terms() const;

	// Adds a variable to the function. This function is called by add_term
	// if the variable needs to be added.
	void add_variable(double* variable, int dimension);

	// Adds a variable to the function, with a change of variables.
	// This can be called on an existing variable to add a change
	// of variables.
	template<typename Change>
	void add_variable_with_change(double* variable,
	                              int dimension)
	{
		add_variable_internal(variable, dimension,
			 std::make_shared<AutoDiffChangeOfVariables<Change>>(new Change));
	}
	template<typename Change, typename T1>
	void add_variable_with_change(double* variable,
	                              int dimension,
	                              T1&& t1)
	{
		add_variable_internal(variable, dimension,
			std::make_shared<AutoDiffChangeOfVariables<Change>>(
				new Change(std::forward<T1>(t1))
			)
		);
	}
	template<typename Change, typename T1, typename T2>
	void add_variable_with_change(double* variable,
	                              int dimension,
	                              T1&& t1, T2&& t2)
	{
		add_variable_internal(variable, dimension,
			std::make_shared<AutoDiffChangeOfVariables<Change>>(
				new Change(std::forward<T1>(t1), std::forward<T2>(t2))
			)
		);
	}
	template<typename Change, typename T1, typename T2, typename T3>
	void add_variable_with_change(double* variable,
	                              int dimension,
	                              T1&& t1, T2&& t2, T3&& t3)
	{
		add_variable_internal(variable, dimension,
			std::make_shared<AutoDiffChangeOfVariables<Change>>(
				new Change(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3))
			)
		);
	}
	template<typename Change, typename T1, typename T2, typename T3, typename T4>
	void add_variable_with_change(double* variable,
	                              int dimension,
	                              T1&& t1, T2&& t2, T3&& t3, T4&& t4)
	{
		add_variable_internal(variable, dimension,
			std::make_shared<AutoDiffChangeOfVariables<Change>>(
				new Change(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3),
				           std::forward<T4>(t4))
			)
		);
	}
	template<typename Change, typename T1, typename T2, typename T3, typename T4, typename T5>
	void add_variable_with_change(double* variable,
	                              int dimension,
	                              T1&& t1, T2&& t2, T3&& t3, T4&& t4, T5&& t5)
	{
		add_variable_internal(variable, dimension,
			std::make_shared<AutoDiffChangeOfVariables<Change>>(
				new Change(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3),
				           std::forward<T4>(t4), std::forward<T4>(t5))
			)
		);
	}

	// Returns the global index of a variable. This index is used for
	// indexing in gradients and Eigen::VectorXd. For normal use, this
	// index is not needed. Use it when e.g. examining the gradient or
	// Hessian.
	size_t get_variable_global_index(double* variable) const;

	// Sets a variable to be constant. In this case, it will not be
	// part of the optimization problem.
	//
	// NOTE: After calling this function, the global indexing of
	//       variables will change permanently.
	void set_constant(double* variable, bool is_constant);

	// Returns the current number of variables the function contains.
	size_t get_number_of_variables() const;

	// Returns the current number of scalars the function contains.
	// (each variable contains of one or several scalars.)
	size_t get_number_of_scalars() const;

	// Sets the number of threads the Function should use when evaluating.
	// Default: number of cores available.
	void set_number_of_threads(int num);

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

	void write_to_stream(std::ostream& out) const;
	void read_from_stream(std::istream& in, std::vector<double>* user_space, const TermFactory& factory);

private:

	// Present here because it is called by a templated function above.
	void add_variable_internal(double* variable,
	                           int dimension,
	                           std::shared_ptr<ChangeOfVariables> change_of_variables = 0);

	class Implementation;
	// unique_pointer would have been nice, but there are issues
	// with sharing these objects across DLL boundaries in VC++.
	Implementation* impl;
};

}  // namespace spii

#endif
