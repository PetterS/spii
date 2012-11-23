
#include <cstring>
#include <iostream>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>


#include <spii/solver.h>

SolverResults::SolverResults()
{
	exit_condition = NA;
}

Solver::Solver()
{
	this->sparsity_mode = AUTO;
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

	// Determine whether to use sparse representation
	// and matrix factorization.
	bool use_sparsity;
	if (this->sparsity_mode == DENSE) {
		use_sparsity = false;
	}
	else if (this->sparsity_mode == SPARSE) {
		use_sparsity = true;
	}
	else {
		if (n <= 50) {
			use_sparsity = false;
		}
		else {
			use_sparsity = true;
		}
	}

	// Current point, gradient and Hessian.
	double fval, fprev, normg0, normdx;
	Eigen::VectorXd x, g;
	Eigen::MatrixXd H;
	Eigen::SparseMatrix<double> sparse_H;
	if (use_sparsity) {
		// Create sparsity pattern for H.
		function.create_sparse_hessian(&sparse_H);
	}

	// Copy the user state to the current point.
	function.copy_user_to_global(&x);
	Eigen::VectorXd x2(n);

	// p will store the search direction.
	Eigen::VectorXd p(function.get_number_of_scalars());

	// Dense Cholesky factorizer.
	Eigen::LLT<Eigen::MatrixXd>* factorization = 0;
	// Sparse Cholesky factorizer.
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double> >* sparse_factorization = 0;
	if (!use_sparsity) {
		factorization = new Eigen::LLT<Eigen::MatrixXd>(n);
	}
	else {
		sparse_factorization = new Eigen::SimplicialLLT<Eigen::SparseMatrix<double> >;
	}

	// Starting value for alpha during line search.
	double alpha_start = 1.0;

	//
	// START MAIN ITERATION
	//
	results->exit_condition = SolverResults::NO_CONVERGENCE;
	for (int iter = 0; iter < this->maximum_iterations; ++iter) {

		// Evaluate function and derivatives.
		if (use_sparsity) {
			fval = function.evaluate(x, &g, &sparse_H);
		}
		else {
			fval = function.evaluate(x, &g, &H);
		}

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

		double e = 0;
		if (!use_sparsity) {
			// Compute smallest eigenvalue of the Hessian.
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ES(H, Eigen:: EigenvaluesOnly);
			Eigen::VectorXd values = ES.eigenvalues();
			e = values.minCoeff();
		}

		// Attempt repeated Cholesky factorization until the Hessian
		// becomes positive semidefinite.
		int factorizations = 0;
		double tau;
		double beta = 1.0;

		Eigen::VectorXd dH;
		if (use_sparsity) {
			dH = sparse_H.diagonal();
		}
		else {
			dH = H.diagonal();
		}
		double mindiag = dH.minCoeff();
		
		if (mindiag > 0) {
			tau = 0;
		}
		else {
			tau = -mindiag + beta;
		}
		while (true) {
			// Add tau*I to the Hessian.
			for (size_t i = 0; i < n; ++i) {
				if (use_sparsity) {
					sparse_H.coeffRef(i, i) = dH(i) + tau;
				}
				else {
					H(i, i) = dH(i) + tau;
				}
			}
			// Attempt Cholesky factorization.
			bool success;
			if (use_sparsity) {
				sparse_factorization->compute(sparse_H);
				success = sparse_factorization->info() == Eigen::Success;
			}
			else {
				factorization->compute(H);
				success = factorization->info() == Eigen::Success;
			}
			factorizations++;
			// Check for success.
			if (success) {
				break;
			}
			tau = std::max(2*tau, beta);
		}

		// Search direction.
		if (use_sparsity) {
			p = sparse_factorization->solve(-g);
		}
		else {
			p = factorization->solve(-g);
		}

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
			char str[1024];
			if (use_sparsity) {
				if (iter == 0) {
					double nnz = double(sparse_H.nonZeros()) / double(n * n);
					std::sprintf(str, "H has %.5f%% non-zeroes.", 100.0 * nnz);
					this->log_function(str);
					this->log_function("Itr      f       max|g_i|     alpha    fac ");
				}
				std::sprintf(str, "%4d %.3e %.3e %.3e %3d",
					iter, fval, normg, alpha, factorizations);
			}
			else {
				double detH = H.determinant();
				double normH = H.norm();

				if (iter == 0) {
					this->log_function("Itr      f       max|g_i|     ||H||      det(H)       e         alpha     fac ");
				}
				std::sprintf(str, "%4d %.3e %.3e %.3e %+.3e %+.3e %.3e %3d",
					iter, fval, normg, normH, detH, e, alpha, factorizations);
			}
			this->log_function(str);
		}
	}

	if (this->log_function) {
		char str[1024];
		std::sprintf(str, " end %.3e %.3e", fval, g.norm());
		this->log_function(str);
	}

	if (factorization) {
		delete factorization;
	}

	if (sparse_factorization) {
		delete sparse_factorization;
	}

	function.copy_global_to_user(x);
}


void cerr_log_function(const std::string& log_message)
{
	std::cerr << log_message << std::endl;
}
