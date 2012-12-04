
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <spii/spii.h>
#include <spii/solver.h>

namespace spii {

void Solver::solve_newton(const Function& function,
                   SolverResults* results) const
{
	double global_start_time = wall_time();

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
	double fval   = std::numeric_limits<double>::quiet_NaN();;
	double fprev  = std::numeric_limits<double>::quiet_NaN();
	double normg0 = std::numeric_limits<double>::quiet_NaN();
	double normg  = std::numeric_limits<double>::quiet_NaN();
	double normdx = std::numeric_limits<double>::quiet_NaN();

	Eigen::VectorXd x, g;
	Eigen::MatrixXd H;
	Eigen::SparseMatrix<double> sparse_H;
	if (use_sparsity) {
		// Create sparsity pattern for H.
		function.create_sparse_hessian(&sparse_H);
		if (this->log_function) {
			double nnz = double(sparse_H.nonZeros()) / double(n * n);
			char str[1024];
			std::sprintf(str, "H is %dx%d with %d (%.5f%%) non-zeroes.",
				sparse_H.rows(), sparse_H.cols(), sparse_H.nonZeros(), 100.0 * nnz);
			this->log_function(str);
		}
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
		// The sparsity pattern of H is always the same. Therefore, it is enough
		// to analyze it once.
		sparse_factorization->analyzePattern(sparse_H);
	}

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
		if (use_sparsity) {
			fval = function.evaluate(x, &g, &sparse_H);
			//std::cerr << sparse_H << std::endl;
			//std::cerr << sparse_H.nonZeros() << std::endl;
		}
		else {
			fval = function.evaluate(x, &g, &H);
			//std::cerr << H << std::endl;
		}

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
		// Attempt repeated Cholesky factorization until the Hessian
		// becomes positive semidefinite.
		//
		start_time = wall_time();

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
			if (tau > 0) {
				for (size_t i = 0; i < n; ++i) {
					if (use_sparsity) {
						int ii = static_cast<int>(i);
						sparse_H.coeffRef(ii, ii) = dH(i) + tau;
					}
					else {
						H(i, i) = dH(i) + tau;
					}
				}
			}
			// Attempt Cholesky factorization.
			bool success;
			if (use_sparsity) {
				sparse_factorization->factorize(sparse_H);
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

			if (factorizations > 100) {
				throw std::runtime_error("Solver::solve: factorization failed.");
			}
		}

		results->matrix_factorization_time += wall_time() - start_time;

		//
		// Solve linear system to obtain search direction.
		//
		start_time = wall_time();

		if (use_sparsity) {
			p = sparse_factorization->solve(-g);
		}
		else {
			p = factorization->solve(-g);
		}

		results->linear_solver_time += wall_time() - start_time;

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
			if (use_sparsity) {
				if (iter == 0) {
					this->log_function("Itr      f       max|g_i|   alpha    fac    tau   min(H_ii)");
				}
				std::sprintf(str, "%4d %+.3e %.3e %.3e %3d   %.1e %+.2e",
					iter, fval, normg, alpha, factorizations, tau, mindiag);
			}
			else {
				double detH = H.determinant();
				double normH = H.norm();

				if (iter == 0) {
					this->log_function("Itr      f       max|g_i|   ||H||     det(H)     alpha    fac  min(H_ii) ");
				}
				std::sprintf(str, "%4d %+.3e %.3e %.3e %+.3e %.3e %3d   %+.2e",
					iter, fval, normg, normH, detH, alpha, factorizations, mindiag);
			}
			this->log_function(str);
		}
		results->log_time += wall_time() - start_time;

		fprev = fval;
		iter++;
	}

	function.copy_global_to_user(x);
	results->total_time = wall_time() - global_start_time;

	if (this->log_function) {
		char str[1024];
		std::sprintf(str, " end %+.3e %.3e", fval, normg);
		this->log_function(str);
	}

	if (factorization) {
		delete factorization;
	}

	if (sparse_factorization) {
		delete sparse_factorization;
	}
}


void cerr_log_function(const std::string& log_message)
{
	std::cerr << log_message << std::endl;
}

}  // namespace spii
