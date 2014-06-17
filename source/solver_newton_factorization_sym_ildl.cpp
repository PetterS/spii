// Petter Strandmark 2013.

#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <spii/spii.h>
#include <spii/solver.h>

#ifndef USE_SYM_ILDL

void spii::Solver::BKP_dense_sym_ildl(const Eigen::MatrixXd& Hinput,
                                      const Eigen::VectorXd& g,
                                      Eigen::VectorXd* p,
                                      SolverResults* results) const
{
	throw std::runtime_error("sym-ildl is not available.");
}

#else

#include <lilc_matrix.h>

#include <spii/sym-ildl-conversions.h>

namespace spii {

void Solver::BKP_dense_sym_ildl(const Eigen::MatrixXd& Hinput,
                                const Eigen::VectorXd& g,
                                Eigen::VectorXd* p,
                                SolverResults* results) const
{
	using namespace std;
	using namespace Eigen;
	double start_time = wall_time();

	//
	// Create sym-ildl matrix.
	//
	auto n = Hinput.rows();
	lilc_matrix<double> Hlilc;
	eigen_to_lilc(Hinput, &Hlilc);

	//
	// Factorize the matrix.
	//
	lilc_matrix<double> Llilc;	          // The lower triangular factor of A.
	vector<int> perm;	                  // A permutation vector containing all permutations on A.
	perm.reserve(Hlilc.n_cols());
	block_diag_matrix<double> Dblockdiag; // The diagonal factor of A.
	Hlilc.sym_equil();
	Hlilc.sym_rcm(perm);
	Hlilc.sym_perm(perm);
	const double fill_factor = 1.0;
	const double tol         = 1e-12;
	Hlilc.ildl(Llilc, Dblockdiag, perm, fill_factor, tol, 1.0);

	//
	// Convert back to Eigen matrices.
	//
	MyPermutation P(perm);
	auto L = lilc_to_eigen(Llilc);
	auto D = block_diag_to_eigen(Dblockdiag);
	auto S = diag_to_eigen(Hlilc.S);

	//
	// Modify the block diagonalization.
	//
	const double delta = 1e-12;
	MatrixXd B(n, n);
	B.setZero();

	bool onebyone;
	for (int i = 0; i < n; i = onebyone ? i+1 : i+2 ) {
		onebyone = (i == n-1 || D(i+1, i) == 0.0);

		if ( onebyone ) {
		    B(i, i) = std::max(D(i, i), delta);
		}
		else {
		    auto block_a = D(i, i);
		    auto block_b = D(i+1, i+1);
		    auto block_d = D(i+1, i);
			auto lambda = (block_a+block_d)/2.0 - std::sqrt(4.0*block_b*block_b + (block_a - block_d)*(block_a - block_d))/2.0;
			B.block(i, i, 2, 2) = D.block(i, i, 2, 2);
			B(i, i)     += lambda + delta;
			B(i+1, i+1) += lambda + delta;
		}
	}

	results->matrix_factorization_time += wall_time() - start_time;
	//
	// Solve the system.
	//
	start_time = wall_time();

	solve_system_ildl_dense(B, L, S, P, -g, p);

	results->linear_solver_time += wall_time() - start_time;
}

}  // namespace spii
#endif // #ifndef USE_SYM_ILDL
