
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

		// If alpha is negative, modify H so that it becomes positive definite.
		if (e > 1e-6) {
			// Matrix is already positive definite.
		}
		else if (e < -1e-6) {
			// Modify Hessian in order to obtain a positive definite matrix.
			for (size_t i = 0; i < n; ++i) {
				H(i, i) += -1.5 * e;
			}
		}
		else {
			// Hessian is (close to) singular.
			this->log_function("H (close to) singular.");
			break;
		}
		
		/*
		// Factorize the modified Hessian (now positive definite).
		Eigen::LLT<Eigen::MatrixXd> factorization(H);
		*/

		// Factorize the Hessian with a robust Cholesky algorithm.
		Eigen::LDLT<Eigen::MatrixXd> factorization(H);

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
				this->log_function("Itr      f        ||g||       ||H||      det(H)       e         alpha ");
			}
			char str[1024];
			std::sprintf(str, "%4d %.3e %.3e %.3e %+.3e %+.3e %.3e", iter, fval, normg, H.norm(), H.determinant(), e, alpha);
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
