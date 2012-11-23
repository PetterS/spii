
#include <cstring>
#include <iostream>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <spii/solver.h>

Solver::Solver()
{
	this->log_function = cerr_log_function;
	this->maximum_iterations = 100;
}

void Solver::Solve(const Function& function,
                   SolverResults* results) const
{
	// Dimension of problem.
	size_t n = function.get_number_of_scalars();

	// Current point, gradient and Hessian.
	double fval;
	Eigen::VectorXd x, g;
	Eigen::MatrixXd H;
	// Copy the user state to the current point.
	function.copy_user_to_global(&x);
	Eigen::VectorXd x2(n);

	Eigen::VectorXd p(function.get_number_of_scalars());

	for (int iter = 0; iter < this->maximum_iterations; ++iter) {
		// Evaluate function and derivatives.
		fval = function.evaluate(x, &g, &H);
		double normg = g.norm();

		if (normg < 1e-9) {
			break;
		}

		if (fval != fval) {
			// NaN encountered.
			if (this->log_function) {
				this->log_function("f(x) is NaN.");
			}
			break;
		}

		if (fval ==  std::numeric_limits<double>::infinity() ||
			fval == -std::numeric_limits<double>::infinity()) {
			// Infinity encountered.
			if (this->log_function) {
				this->log_function("f(x) is infinity.");
			}
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

		double alpha = 1.0;
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

		if (this->log_function) {
			if (iter == 0) {
				this->log_function("Itr      f        ||g||       ||H||      det(H)       e         alpha     fac ");
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
