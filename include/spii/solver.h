// Petter Strandmark 2012–2013.
#ifndef SPII_SOLVER_H
#define SPII_SOLVER_H
// The Solver class is a lightweight class defining settings
// for a solver.
//
// The member function Solver::solve uses minimizes a Function
// using the settings in the Solver.
//

#include <functional>
#include <iostream>
#include <string>

#include <spii/spii.h>
#include <spii/function.h>

namespace spii {

// SolverResults contains the result of a call to Solver::solve.
struct SPII_API SolverResults
{

	// The exit condition specifies how the solver terminated.
	enum {GRADIENT_TOLERANCE, // Gradient tolerance reached.
	      FUNCTION_TOLERANCE, // Function improvement tolerance reached.
	      ARGUMENT_TOLERANCE, // Argument improvement tolerance reached.
	      NO_CONVERGENCE,     // Maximum number of iterations reached.
	      FUNCTION_NAN,       // Nan encountered.
	      FUNCTION_INFINITY,  // Infinity encountered.
	      INTERNAL_ERROR,     // Internal error.
	      NA} exit_condition = NA;

	// Returns true if the exit_condition indicates convergence.
	bool exit_success() const
	{
		return exit_condition == GRADIENT_TOLERANCE ||
		       exit_condition == FUNCTION_TOLERANCE ||
		       exit_condition == ARGUMENT_TOLERANCE;
	}

	double startup_time                = 0;
	double function_evaluation_time    = 0;
	double stopping_criteria_time      = 0;
	double matrix_factorization_time   = 0;
	double lbfgs_update_time           = 0;
	double linear_solver_time          = 0;
	double backtracking_time           = 0;
	double log_time                    = 0;
	double total_time                  = 0;

	// The minimum value of the function being minimized is
	// in this interval. These members are only set by global
	// optmization solvers.
	double optimum_lower = - std::numeric_limits<double>::infinity();
	double optimum_upper =   std::numeric_limits<double>::infinity();
};

SPII_API std::ostream& operator<<(std::ostream& out, const SolverResults& results);

struct FactorizationCacheInternal;
class FactorizationCache
{
public:
	FactorizationCache(int n);
	~FactorizationCache();
	FactorizationCacheInternal* data;
};

#ifdef _WIN32
	SPII_API_EXTERN_TEMPLATE template class SPII_API std::function<void(const std::string&)>;
#endif

class SPII_API Solver
{
public:
	Solver();
	virtual ~Solver();

	virtual void solve(const Function& function, SolverResults* results) const = 0;

	// Function called each iteration with a log message.
	// Default: print to std::cerr.
	std::function<void(const std::string& log_message)> log_function;

	// Maximum number of iterations.
	int maximum_iterations = 100;

	// Gradient tolerance. The solver terminates if
	// ||g|| / ||g0|| < tol, where ||.|| is the maximum
	// norm.
	double gradient_tolerance = 1e-12;

	// Function improvement tolerance. The solver terminates
	// if |df| / (|f| + tol) < tol.
	double function_improvement_tolerance = 1e-12;

	// Argument improvement tolerance. The solver terminates
	// if ||dx|| / (||x|| + tol) < tol.
	double argument_improvement_tolerance = 1e-12;

	// The line search is completed when
	//   f(x + alpha * p) <= f(x) + c * alpha * gTp.
	// In each iteration, alpha *= rho.
	double line_search_c   = 1e-4;
	double line_search_rho = 0.5;

protected:

	// Computes a Newton step given a function, a gradient and a
	// Hessian.
	bool check_exit_conditions(const double fval,
	                           const double fprev,
	                           const double gnorm,
	                           const double gnorm0,
	                           const double xnorm,
	                           const double dxnorm,
	                           const bool last_iteration_successful,
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

	// Performs a BKP block diagonal factorization, modifies it, and
	// solvers the linear system.
	void BKP_dense(const Eigen::MatrixXd& H,
	               const Eigen::VectorXd& g,
	               const FactorizationCache& cache,
	               Eigen::VectorXd* p,
	               SolverResults* results) const;
};

// Newton's method. It requires first and
// second-order derivatives. Generally converges
// quickly. It is slow and requires a lot of
// memory if the Hessian is dense.
class SPII_API NewtonSolver
	: public Solver
{
public:
	// Mode of operation. How the Hessian is stored.
	// Default: AUTO.
	enum {DENSE, SPARSE, AUTO} sparsity_mode = AUTO;

	// The default factorization method is the BKP block
	// diagonal modification (Nocedal and Wright, p. 55).
	// Alternatively, it is possible to use iterative diagonal
	// modification of the Hessian. This is also used for
	// sparse systems.
	enum {BKP, ITERATIVE} factorization_method = BKP;

	virtual void solve(const Function& function, SolverResults* results) const;
};

// L-BFGS. Requires only first-order derivatives
// and generally converges quickly. Always uses
// relatively little memory.
class SPII_API LBFGSSolver
	: public Solver
{
public:
	// Number of vectors L-BFGS should save in its history.
	int lbfgs_history_size = 10;

	// If the relative function improvement is less than this
	// value, L-BFGS will discard its history and restart.
	double lbfgs_restart_tolerance = 1e-6;

	virtual void solve(const Function& function, SolverResults* results) const;
};

// Nelder-Mead requires no derivatives. It generally
// produces slightly more inaccurate solutions in many
// more iterations.
class SPII_API NelderMeadSolver
	: public Solver
{
public:
	// Area tolerance. The solver terminates if
	// ||a|| / ||a0|| < tol, where ||.|| is the maximum
	// norm.
	double area_tolerance = 1e-12;

	// Length tolerance. The solver terminates if
	// ||a|| / ||a0|| < tol, where ||.|| is the maximum
	// norm.
	double length_tolerance = 1e-12;

	virtual void solve(const Function& function, SolverResults* results) const;
};

// For most problems, there is no reason to choose
// pattern search over Nelder-Mead.
class SPII_API PatternSolver
	: public Solver
{
public:
	// Area tolerance. The solver terminates if
	// ||a|| / ||a0|| < tol, where ||.|| is the maximum
	// norm.
	double area_tolerance = 1e-12;

	virtual void solve(const Function& function, SolverResults* results) const;
};

// (Experimental) Global optimization using interval
// arithmetic.
class SPII_API GlobalSolver
	: public Solver
{
public:
	void solve_global(const Function& function,
	                  const IntervalVector& start_box,
	                  SolverResults* results) const;
	
	// Does not do anything. The global solver requires the
	// extended interface above.
	virtual void solve(const Function& function, SolverResults* results) const;
};

}  // namespace spii

#endif
