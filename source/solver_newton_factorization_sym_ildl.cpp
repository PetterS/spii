// Petter Strandmark 2013.

#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

// GNU 4.8.1 define _X on Cygwin.
// This breaks Eigen.
// http://eigen.tuxfamily.org/bz/show_bug.cgi?id=658
#ifdef _X
#undef _X
#endif
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
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

	// The (very challenging) NIST test suite fails if eps
	// is kept at its default value when calling sym_equil.
	auto prev_eps = Hlilc.eps;
	Hlilc.eps = 0;
	Hlilc.sym_equil();
	Hlilc.eps = prev_eps;

	Hlilc.sym_rcm(perm);
	Hlilc.sym_perm(perm);
	const double fill_factor = 1.0;
	const double tol         = 1e-12;
	const double pp_tol      = 1.0; // For full Bunch-Kaufman.
	Hlilc.ildl(Llilc, Dblockdiag, perm, fill_factor, tol, pp_tol);

	//
	// Convert back to Eigen matrices.
	//
	MyPermutation P(perm);
	auto L = lilc_to_eigen(Llilc);
	auto B = block_diag_to_eigen(Dblockdiag);
	auto S = diag_to_eigen(Hlilc.S);

	//
	// Modify the block diagonalization.
	//
	const double delta = 1e-12;

	MatrixXd Q(n, n);
	VectorXd tau(n);
	VectorXd lambda(n);
	Q.setZero();

	SelfAdjointEigenSolver<MatrixXd> eigensolver;

	bool onebyone;
	for (int i = 0; i < n; i = onebyone ? i+1 : i+2 ) {
		onebyone = (i == n-1 || B(i+1, i) == 0.0);

		if ( onebyone ) {
			lambda(i) = B(i, i);
			if (lambda(i) >= delta) {
				tau(i) = 0;
			}
			else {
				tau(i) = delta - (1.0 + delta) * lambda(i);
			}
			Q(i, i) = 1;
		}
		else {
			spii_assert(B(i+1, i) == B(i, i+1));

			eigensolver.compute(B.block(i, i, 2, 2));
			lambda(i)   = eigensolver.eigenvalues()(0);
			lambda(i+1) = eigensolver.eigenvalues()(1);
			for (int k = i; k <= i + 1; ++k) {
				if (lambda(k) >= delta) {
					tau(k) = 0;
				}
				else {
					tau(k) = delta - (1.0 + delta) * lambda(k);
				}
			}

			Q.block(i, i, 2, 2) = eigensolver.eigenvectors();
		}
	}

	B = B + Q * tau.asDiagonal() * Q.transpose();

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
