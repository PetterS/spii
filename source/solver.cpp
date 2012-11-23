
#include <cstring>
#include <iostream>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <spii/solver.h>

SolverResults::SolverResults()
{
	exit_condition = NA;
}

Solver::Solver()
{
	this->log_function = cerr_log_function;
	this->maximum_iterations = 100;
	this->gradient_tolerance = 1e-12;
	this->function_improvement_tolerance = 1e-12;
	this->argument_improvement_tolerance = 1e-12;
}

void Solver::Solve(const Function& function,
                   SolverResults* results) const
{
	// Dimension of problem.
	size_t n = function.get_number_of_scalars();

	// Current point, gradient and Hessian.
	double fval, fprev, normg0, normdx;
	Eigen::VectorXd x, g;
	Eigen::MatrixXd H;
	// Copy the user state to the current point.
	function.copy_user_to_global(&x);
	Eigen::VectorXd x2(n);

	Eigen::VectorXd p(function.get_number_of_scalars());

	// Starting value for alpha during line search.
	double alpha_start = 1.0;

	for (int iter = 0; iter < this->maximum_iterations; ++iter) {
		// Evaluate function and derivatives.
		fval = function.evaluate(x, &g, &H);
		double normg = std::max(g.maxCoeff(), -g.minCoeff());
		if (iter == 0) {
			normg0 = normg;
		}

		if (fval != fval) {
			// NaN encountered.
			if (this->log_function) {
				this->log_function("f(x) is NaN.");
			}
			results->exit_condition = SolverResults::NAN;
			break;
		}

		if (iter >= 1) {
			if (normg / normg0 < this->gradient_tolerance) {
				if (this->log_function) {
					this->log_function("Gradient tolerance.");
				}
				results->exit_condition = SolverResults::GRADIENT_TOLERANCE;
				break;
			}

			if (abs(fval - fprev) / (abs(fval) + this->function_improvement_tolerance) <
			                                     this->function_improvement_tolerance) {
				if (this->log_function) {
					this->log_function("Function improvement tolerance.");
				}
				results->exit_condition = SolverResults::FUNCTION_TOLERANCE;
				break;
			}

			if (normdx / (x.norm() + this->argument_improvement_tolerance) <
			                         this->argument_improvement_tolerance) {
				if (this->log_function) {
					this->log_function("Variable tolerance.");
				}
				results->exit_condition = SolverResults::ARGUMENT_TOLERANCE;
				break;
			}
		}

		if (fval ==  std::numeric_limits<double>::infinity() ||
			fval == -std::numeric_limits<double>::infinity()) {
			// Infinity encountered.
			if (this->log_function) {
				this->log_function("f(x) is infinity.");
			}
			results->exit_condition = SolverResults::INFINITY;
			break;
		}

		
		// Compute smallest eigenvalue of the Hessian.
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ES(H, Eigen:: EigenvaluesOnly);
		Eigen::VectorXd values = ES.eigenvalues();
		double e = values.minCoeff();

		// Attempt repeated Cholesky factorization until the Hessian
		// becomes positive semidefinite.
		Eigen::LLT<Eigen::MatrixXd> factorization(n);
		int factorizations = 0;
		Eigen::VectorXd dH = H.diagonal();

		double tau;
		double beta = 1.0;
		double mindiag = H.diagonal().minCoeff();
		if (mindiag > 0) {
			tau = 0;
		}
		else {
			tau = -mindiag + beta;
		}
		while (true) {
			// Add tau*I to the Hessian.
			for (size_t i = 0; i < n; ++i) {
				H(i, i) = dH(i) + tau;
			}
			// Attempt Cholesky factorization.
			factorization.compute(H);
			factorizations++;
			// Check for success.
			if (factorization.info() == Eigen::Success) {
				break;
			}
			tau = std::max(2*tau, beta);
		}

		// Search direction.
		p = factorization.solve(-g);

		// Perform back-tracking line search.
		double alpha = alpha_start;
		double rho = 0.5;
		double c = 0.5;
		while (true) {
			x2 = x + alpha * p;
			double lhs = function.evaluate(x2);
			double rhs = fval + c * alpha * g.dot(p);
			if (lhs > rhs) {
				alpha *= rho;
			}
			else {
				break;
			}
		}
		x = x2;

		// Record length of this step.
		normdx = alpha * p.norm();

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
			if (iter == 0) {
				this->log_function("Itr      f       max|g_i|     ||H||      det(H)       e         alpha     fac ");
			}
			char str[1024];
			std::sprintf(str, "%4d %.3e %.3e %.3e %+.3e %+.3e %.3e %3d",
				iter, fval, normg, H.norm(), H.determinant(), e, alpha, factorizations);
			this->log_function(str);
		}
	}

	if (this->log_function) {
		char str[1024];
		std::sprintf(str, " end %.3e %.3e", fval, g.norm());
		this->log_function(str);
	}

	function.copy_global_to_user(x);
}


void cerr_log_function(const std::string& log_message)
{
	std::cerr << log_message << std::endl;
}
