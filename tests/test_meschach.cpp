
#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <Eigen/Dense>

extern "C" {
	#include "matrix.h"
	#include "matrix2.h"
	#include "sparse2.h"
	#undef min
	#undef max
	#undef catch
}

template<typename EigenMat>
MAT* Eigen_to_Meschach(const EigenMat& eigen_matrix)
{
	int m = eigen_matrix.rows();
	int n = eigen_matrix.cols();

	auto A = m_get(m, n);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			A->me[i][j] = eigen_matrix(i, j);
		}
	}

	return A;
}

TEST_CASE("BKP-dense", "")
{
	using namespace std;
	using namespace Eigen;

	int n = 4;

	MatrixXd Amat(4, 4);
	Amat.row(0) << 1, 2, 3, 1;
	Amat.row(1) << 2, 6, 1, 8;
	Amat.row(2) << 9, 8, 7, 6;
	Amat.row(3) << 9, 9, 1, 6;

	auto A = Eigen_to_Meschach(Amat);
	m_foutput(stderr, A);

	PERM* pivot  = px_get(4);
	PERM* block = px_get(4);
	BKPfactor(A, pivot, block);

	m_foutput(stderr, A);
	px_foutput(stderr, block);
	px_foutput(stderr, pivot);


	cerr << endl << endl;

	MatrixXd B(4, 4);
	MatrixXd Q(4, 4);
	VectorXd tau(4);
	VectorXd lambda(4);
	B.setZero();
	Q.setZero();

	SelfAdjointEigenSolver<MatrixXd> eigensolver;

	double delta = 1e-10;

	int onebyone;
	for (int i = 0; i < n; i = onebyone ? i+1 : i+2 ) {
		onebyone = ( block->pe[i] == i );
		if ( onebyone ) {
		    B(i, i) = m_entry(A,i,i);
			lambda(i) = B(i, i);
			if (lambda(i) >= delta) {
				tau(i) = 0;
			}
			else {
				tau(i) = delta - lambda(i);
			}
			Q(i, i) = 1;
		}
		else {
		    auto a11 = m_entry(A,i,i);
		    auto a22 = m_entry(A,i+1,i+1);
		    auto a12 = m_entry(A,i+1,i);
			B(i,   i)   = a11;
			B(i+1, i)   = a12;
			B(i,   i+1) = a12;
			B(i+1, i+1) = a22;
			eigensolver.compute(B.block(i, i, 2, 2));

			lambda(i)   = eigensolver.eigenvalues()(0);
			lambda(i+1) = eigensolver.eigenvalues()(1);
			for (int k = i; k <= i + 1; ++k) {
				if (lambda(k) >= delta) {
					tau(k) = 0;
				}
				else {
					tau(k) = delta - lambda(k);
				}
			}

			Q.block(i, i, 2, 2) = eigensolver.eigenvectors();
		}
	}
	cerr << "B = \n" << B << endl << endl;
	cerr << "lambda = " << tau.transpose() << endl << endl;
	cerr << "tau = " << tau.transpose() << endl << endl;
	cerr << "Q = \n" << Q << endl << endl;
	cerr << "Q*lambda*Q^T = \n" << Q * lambda.asDiagonal() * Q.transpose() << endl << endl;

	// Check that the block-wise eigendecomposition was correct.
	CHECK(((B - Q * lambda.asDiagonal() * Q.transpose()).norm()) < 1e-10);

	MatrixXd F = Q * tau.asDiagonal() * Q.transpose();
	cerr << "F = Q*tau*Q^T = \n" << F << endl << endl;
	cerr << "B + F = \n" << B + F << endl << endl;


	m_free(A);
	px_free(pivot);
	px_free(block);
}

/*
TEST_CASE("BKP-sparse", "")
{
	int m = 5;
	int n = 5;
	int deg = 5;
	SPMAT* A = sp_get(m, n, deg);

	int I[25]    = {1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5};
	int J[25]    = {1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5};
	double V[25] = {1,7,3,5,1,5,2,6,6,7,4,1,2,8,1,7,5,9,1,4,1,9,0,7,7};

	for (int i = 0; i < 25; ++i) {
		if (V[i] != 0) {
			sp_set_val(A, I[i] - 1, J[i] - 1, V[i]);
		}
	}

	sp_foutput2(stderr, A);

	SPMAT* B = sp_copy(A);
	PERM* pivot  = px_get(5);
	PERM* blocks = px_get(5);
	spBKPfactor(B, pivot, blocks, 1e-16);

	sp_fo1utput2(stderr, B);
	px_foutput(stderr, pivot);
	px_foutput(stderr, blocks);

	SP_FREE(A);
	SP_FREE(B);
}
*/