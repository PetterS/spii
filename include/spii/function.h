// Petter Strandmark 2012.
#ifndef SPII_FUNCTION_H
#define SPII_FUNCTION_H
// This header defines the Function class which is used
// to store an objective function to be optimized.

#include <cstddef>
#include <map>
#include <set>
using std::size_t;

#include <Eigen/SparseCore>

#include <spii/term.h>

namespace spii {

// These two structs are used by Function to store added
// variables and terms.
struct AddedVariable
{
	int dimension;
	size_t global_index;
	mutable std::vector<double>  temp_space;
};
struct AddedTerm
{
	const Term* term;
	std::vector<AddedVariable*> user_variables;
	// Temporary storage for a point and hessian.
	mutable std::vector<double*> temp_variables;
	mutable std::vector< std::vector<Eigen::MatrixXd> > hessian;
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

	// Returns the current number of variables the function contains.
	size_t get_number_of_variables() const
	{
		return variables.size();
	}

	// Returns the current number of scalars the function contains.
	// (each variable contains of one or several scalars.)
	size_t get_number_of_scalars() const
	{
		return number_of_scalars;
	}

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
	size_t get_number_of_terms() const
	{
		return terms.size();
	}

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

protected:
	// Copies variables from a global vector x to the Function's
	// local storage.
	void copy_global_to_local(const Eigen::VectorXd& x) const;
	// Copies variables from a global vector x to the storage
	// provided by the user.
	void copy_global_to_user(const Eigen::VectorXd& x) const;
	// Copies variables from a the storage provided by the user
	// to a global vector x.
	void copy_user_to_global(Eigen::VectorXd* x) const;

	// A set of all terms added to the function. This is
	// used when the function is destructed.
	std::set<const Term*> added_terms;

	// All variables added to the function.
	std::map<double*, AddedVariable> variables;

	// Each variable can have several dimensions. This member
	// keeps track of the total number of scalars.
	size_t number_of_scalars;

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
};

}  // namespace spii

#endif
