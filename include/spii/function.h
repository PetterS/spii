#ifndef SPII_FUNCTION_H
#define SPII_FUNCTION_H

#include <cstddef>
#include <map>
using std::size_t;

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
	Function();

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

protected:
	size_t global_index(double* variable) const;
	void copy_global_to_local(const Eigen::VectorXd& x) const;
	void copy_user_to_global(Eigen::VectorXd* x) const;

	// Has to be mutable because the temporary storage
	// needs to be written to.
	mutable std::map<double*, AddedVariable> variables;
	size_t number_of_scalars;

	// Has to be mutable because the temporary storage
	// needs to be written to.
	mutable std::vector<AddedTerm> terms;
};

#endif SPII_FUNCTION_H
