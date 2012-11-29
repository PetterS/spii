// Petter Strandmark 2012.
#ifndef SPII_SOLVER_H
#define SPII_SOLVER_H
// The Solver class is a lightweight class defining settings
// for a solver.
//
// The member function Solver::solve uses minimizes a Function
// using the settings in the Solver.
//

#include <iostream>
#include <string>

#include <spii/function.h>

namespace spii {

// SolverResults contains the result of a call to Solver::solve.
struct SolverResults
{
	SolverResults();

	// The exit condition specifies how the solver terminated.
	enum {GRADIENT_TOLERANCE, // Gradient tolerance reached.
	      FUNCTION_TOLERANCE, // Function improvement tolerance reached.
	      ARGUMENT_TOLERANCE, // Argument improvement tolerance reached.
	      NO_CONVERGENCE,     // Maximum number of iterations reached.
	      FUNCTION_NAN,       // Nan encountered.
	      FUNCTION_INFINITY,  // Infinity encountered.
	      ERROR,              // Internal error.
	      NA} exit_condition;

	double startup_time;
	double function_evaluation_time;
	double stopping_criteria_time;
	double matrix_factorization_time;
	double lbfgs_update_time;
	double linear_solver_time;
	double backtracking_time;
	double log_time;
	double total_time;
};

std::ostream& operator<<(std::ostream& out, const SolverResults& results);

// Default log function that prints its argument to std::cerr.
void cerr_log_function(const std::string& log_message);

class Solver
{
public:
	Solver();
	void Solve(const Function& function,
	           SolverResults* results) const;

	void solve_lbfgs(const Function& function,
	                 SolverResults* results) const;

	// Mode of operation. How the Hessian is stored.
	// Default: AUTO.
	enum {DENSE, SPARSE, AUTO} sparsity_mode;

	// Function called each iteration with a log message.
	// Default: cerr_log_function.
	void (*log_function)(const std::string& log_message);

	// Maximum number of iterations. Default: 100.
	int maximum_iterations;

	// Gradient tolerance. The solver terminates if
	// ||g|| / ||g0|| < tol, where ||.|| is the maximum
	// norm. Default: 1e-12.
	double gradient_tolerance;

	// Function improvement tolerance. The solver terminates
	// if |df| / (|f| + tol) < tol. Default: 1e-12.
	double function_improvement_tolerance;

	// Argument improvement tolerance. The solver terminates
	// if ||dx|| / (||x|| + tol) < tol. Default: 1e-12.
	double argument_improvement_tolerance;

	// Number of vectors L-BFGS should save in its history.
	// Default: 10.
	int lbfgs_history_size;

private:

	// Computes a Newton step given a function, a gradient and a
	// Hessian.
	bool check_exit_conditions(const double fval,
	                           const double fprev,
	                           const double gnorm,
							   const double gnorm0,
	                           const double xnorm,
	                           const double dxnorm,
	                           SolverResults* results) const;

	// Performs a line search from x along direction p. Returns
	// alpha, the multiple of p to get to the new point.
	double perform_linesearch(const Function& function,
	                          const Eigen::VectorXd& x,
	                          const double fval,
	                          const Eigen::VectorXd& g,
	                          const Eigen::VectorXd& p,
	                          Eigen::VectorXd* scratch,
	                          const double start_alpha = 1.0) const;                     
};

#endif

}  // namespace spii
