
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <spii/auto_diff_term.h>
#include <spii/solver.h>

// "An analysis of the behavior of a glass of genetic adaptive systems."
// K.A. De Jong.  Ph.D. thesis, University of Michigan, 1975.
struct GeneralizedRosenbrockTerm
{
	template<typename R>
	R operator()(const R* const x1, const R* const x2) const
	{
		R d0 =  (*x1) * (*x1) - (*x2);
		R d1 =  1 - (*x1);
		return 100 * d0*d0 + d1*d1;
	}
};

// An easier variant.
struct EasyRosenbrockTerm
{
	template<typename R>
	R operator()(const R* const x1, const R* const x2) const
	{
		R d0 =  (*x1) * (*x1) - (*x2);
		R d1 =  1 - (*x1);
		return d0*d0 + d1*d1;
	}
};

template<typename Functor, size_t n, bool lbfgs>
void test_rosenbrock()
{
	Function f;
	std::vector<double> x(n);
	for (size_t i = 0; i < n; ++i) {
		f.add_variable(&x[i], 1);
	}

	// Initial values.
	for (size_t i = 0; i < n; ++i) {
		if (i % 2 == 0) {
			x[i] = -1.2;
		}
		else {
			x[i] = 1.0;
		}
	}

	// Add all diagonal terms.
	for (size_t i = 0; i < n - 1; ++i) {
		f.add_term(new AutoDiffTerm<Functor, 1, 1>
		               (new Functor()), &x[i], &x[i+1]);
	}

	Solver solver;
	solver.maximum_iterations = 10000;
	SolverResults results;
	if (lbfgs) {
		solver.solve_lbfgs(f, &results);
	}
	else {
		solver.Solve(f, &results);
	}
	std::cerr << results;

	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE ||
	            results.exit_condition == SolverResults::FUNCTION_TOLERANCE ||
	            results.exit_condition == SolverResults::GRADIENT_TOLERANCE);

	EXPECT_LT( std::abs(f.evaluate()), 1e-9);

	for (size_t i = 0; i < n - 1; ++i) {
		ASSERT_LT( std::abs(x[i] - 1.0), 1e-9);
	}
}


TEST(Newton, EasyRosenbrock1000)
{
	test_rosenbrock<EasyRosenbrockTerm, 1000, false>();
}

TEST(Newton, EasyRosenbrock10000)
{
	test_rosenbrock<EasyRosenbrockTerm, 10000, false>();
}

TEST(LBFGS, EasyRosenbrock1000)
{
	test_rosenbrock<EasyRosenbrockTerm, 1000, true>();
}

TEST(LBFGS, EasyRosenbrock10000)
{
	test_rosenbrock<EasyRosenbrockTerm, 10000, true>();
}


struct LennardJones
{
	template<typename R>
	R operator()(const R* const p1, const R* const p2) const
	{
		R dx = p1[0] - p2[0];
		R dy = p1[1] - p2[1];
		R dz = p1[2] - p2[2];
		R r2 = dx*dx + dy*dy + dz*dz;
		R r6  = r2*r2*r2;
		R r12 = r6*r6;
		return 1.0 / r12 - 1.0 / r6;
	}
};

template<bool lbfgs, int n>
void test_Lennard_Jones()
{
	std::mt19937 prng(0);
	std::normal_distribution<double> normal;
	auto randn = std::bind(normal, prng);

	Function potential;

	std::vector<Eigen::Vector3d> points(n*n*n);
	for (int x = 0; x < n; ++x) {
		for (int y = 0; y < n; ++y) {
			for (int z = 0; z < n; ++z) {
				points[x + y*n + z*n*n][0] = x + 0.05 * randn();
				points[x + y*n + z*n*n][1] = y + 0.05 * randn();
				points[x + y*n + z*n*n][2] = z + 0.05 * randn();

				potential.add_variable(&points[x + y*n + z*n*n][0], 3);
			}
		}
	}

	// Add all pairwise terms
	for (int i = 0; i < n*n*n; ++i) {
		for (int j = i + 1; j < n*n*n; ++j) {
			potential.add_term(
				new AutoDiffTerm<LennardJones, 3, 3>(
					new LennardJones),
					&points[i][0],
					&points[j][0]);
		}
	}

	Solver solver;
	solver.maximum_iterations = 1000;
	solver.function_improvement_tolerance = 1e-6;

	// All points interact with all points, so the Hessian
	// will be dense.
	solver.sparsity_mode = Solver::DENSE;
	SolverResults results;
	if (lbfgs) {
		solver.solve_lbfgs(potential, &results);
	}
	else {
		solver.Solve(potential, &results);
	}
	std::cerr << results;
	potential.print_timing_information(std::cerr);

	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE ||
	            results.exit_condition == SolverResults::FUNCTION_TOLERANCE ||
	            results.exit_condition == SolverResults::GRADIENT_TOLERANCE);
}

TEST(Newton, LennardJones)
{
	test_Lennard_Jones<false, 5>();
}

TEST(LBFGS, LennardJones)
{
	test_Lennard_Jones<true, 5>();
}

struct Trid1
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R d = *x - 1.0;
		return d*d;
	}
};

struct Trid2
{
	template<typename R>
	R operator()(const R* const x1, const R* const x2) const
	{
		return - (*x1) * (*x2);
	}
};

template<size_t n, bool lbfgs>
void test_trid()
{
	Function f;

	std::vector<double> x(n, 1.0);
	for (size_t i = 0; i < n; ++i) {
		f.add_variable(&x[i], 1);
	}

	for (size_t i = 0; i < n; ++i) {
		f.add_term( new AutoDiffTerm<Trid1, 1>(
			new Trid1),
			&x[i]);
	}

	for (size_t i = 1; i < n; ++i) {
		f.add_term( new AutoDiffTerm<Trid2, 1, 1>(
			new Trid2),
			&x[i],
			&x[i-1]);
	}

	Solver solver;
	solver.maximum_iterations = 100;
	//solver.argument_improvement_tolerance = 1e-16;
	//solver.gradient_tolerance = 1e-16;
	SolverResults results;
	if (lbfgs) {
		solver.maximum_iterations = 1000;
		solver.solve_lbfgs(f, &results);
	}
	else {
		solver.Solve(f, &results);
	}
	std::cerr << results;

	double fval = f.evaluate();
	// Global optimum is 
	//
	//   x[i] = (i + 1) * (n - i)
	//
	// Therefore, it is hard to calulate the optimal function
	// value for large n.
	//
	double tol  = 1e-8;
	if (n >= 10000) {
		tol = 1e-6;
	}
	if (n >= 100000) {
		tol = 1e-4;
	}
	EXPECT_LT(std::abs(fval + n * (n+4) * (n-1) / 6.0) / std::abs(fval), tol);
	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE ||
	            results.exit_condition == SolverResults::FUNCTION_TOLERANCE ||
	            results.exit_condition == SolverResults::GRADIENT_TOLERANCE);

	if (n <= 10) {
		for (size_t i = 0; i < n; ++i) {
			std::cerr << "x[" << i << "] = " << x[i] << '\n';
		}
	}
}

TEST(Newton, Trid10) 
{
	test_trid<10, false>();
}

TEST(Newton, Trid1000) 
{
	test_trid<1000, false>();
}

TEST(Newton, Trid10000) 
{
	test_trid<10000, false>();
}

struct LogBarrier01
{
	template<typename R>
	R operator()(const R* const x) const
	{
		return -log(x[0]) - log(1.0 - x[0]);
	}
};

struct QuadraticFunction1
{
	QuadraticFunction1(double b)
	{
		this->b = b;
	}

	template<typename R>
	R operator()(const R* const x) const
	{
		R d = x[0] - b;
		return d * d;
	}

	double b;
};

TEST(Newton, Barrier)
{
	std::mt19937 prng(0);
	std::normal_distribution<double> normal;
	auto randn = std::bind(normal, prng);

	const int n = 10000;
	Function f;

	std::vector<double> x(n, 0.5);
	for (size_t i = 0; i < n; ++i) {
		f.add_variable(&x[i], 1);
		f.add_term( new AutoDiffTerm<QuadraticFunction1, 1>(
			new QuadraticFunction1(0.5 + randn())),
			&x[i]);
		f.add_term( new AutoDiffTerm<LogBarrier01, 1>(
			new LogBarrier01),
			&x[i]);
	}

	Solver solver;
	solver.maximum_iterations = 100;
	SolverResults results;
	solver.Solve(f, &results);

	std::cerr << results;
	f.print_timing_information(std::cerr);

	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE ||
	            results.exit_condition == SolverResults::FUNCTION_TOLERANCE ||
	            results.exit_condition == SolverResults::GRADIENT_TOLERANCE);
}
