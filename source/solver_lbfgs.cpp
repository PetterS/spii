

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <spii/spii.h>
#include <spii/solver.h>


void Solver::solve_lbfgs(const Function& function,
                          SolverResults* results) const
{
	double global_start_time = wall_time();

	// Dimension of problem.
	size_t n = function.get_number_of_scalars();

	// Current point, gradient and Hessian.
	double fval   = std::numeric_limits<double>::quiet_NaN();;
	double fprev  = std::numeric_limits<double>::quiet_NaN();
	double normg0 = std::numeric_limits<double>::quiet_NaN();
	double normg  = std::numeric_limits<double>::quiet_NaN();
	double normdx = std::numeric_limits<double>::quiet_NaN();

	Eigen::VectorXd x, g;
	Eigen::MatrixXd H;
	Eigen::SparseMatrix<double> sparse_H;
	function.create_sparse_hessian(&sparse_H);

	// Copy the user state to the current point.
	function.copy_user_to_global(&x);
	Eigen::VectorXd x2(n);

	// p will store the search direction.
	Eigen::VectorXd p(function.get_number_of_scalars());

	//
	// START MAIN ITERATION
	//
	results->startup_time   = wall_time() - global_start_time;
	results->exit_condition = SolverResults::ERROR;
	int iter = 0;
	while (true) {

		//
		// Evaluate function and derivatives.
		//
		double start_time = wall_time();
		fval = function.evaluate(x, &g, &sparse_H);

		normg = std::max(g.maxCoeff(), -g.minCoeff());
		if (iter == 0) {
			normg0 = normg;
		}
		results->function_evaluation_time += wall_time() - start_time;

		//
		// Test stopping criteriea
		//
		start_time = wall_time();
		if (this->check_exit_conditions(fval, fprev, normg,
			                            normg0, x.norm(), normdx,
			                            results)) {
			break;
		}
		if (iter >= this->maximum_iterations) {
			results->exit_condition = SolverResults::NO_CONVERGENCE;
			break;
		}
		results->stopping_criteria_time += wall_time() - start_time;

		//
		// Compute search direction
		//
		p = -g;

		//
		// Perform line search.
		//
		start_time = wall_time();
		double alpha = this->perform_linesearch(function, x, fval, g, p, &x2);
		// Record length of this step.
		normdx = alpha * p.norm();
		// Update current point.
		x = x + alpha * p;
		results->backtracking_time += wall_time() - start_time;

		//
		// Log the results of this iteration.
		//
		start_time = wall_time();

		int log_interval = 1;
		if (iter > 30) {
			log_interval = 10;
		}
		if (iter > 200) {
			log_interval = 100;
		}
		if (iter > 2000) {
			log_interval = 1000;
		}
		if (this->log_function && iter % log_interval == 0) {
			char str[1024];
				if (iter == 0) {
					this->log_function("Itr       f       max|g_i|     alpha");
				}
				std::sprintf(str, "%4d %+.3e %.3e %.3e",
					iter, fval, normg, alpha);
			this->log_function(str);
		}
		results->log_time += wall_time() - start_time;

		fprev = fval;
		iter++;
	}

	if (this->log_function) {
		char str[1024];
		std::sprintf(str, " end %.3e %.3e", fval, normg);
		this->log_function(str);
	}

	function.copy_global_to_user(x);

	results->total_time = wall_time() - global_start_time;
}