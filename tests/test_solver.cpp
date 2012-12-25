
#include <cmath>
#include <limits>
#include <random>

#include <gtest/gtest.h>

#include <spii/auto_diff_term.h>
#include <spii/constraints.h>
#include <spii/solver.h>

using namespace spii;

struct Rosenbrock
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R d0 =  x[1] - x[0]*x[0];
		R d1 =  1 - x[0];
		return 100 * d0*d0 + d1*d1;
	}
};

void test_method(Solver::Method method, const Solver& solver)
{
	Function f;
	double x[2] = {-1.2, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Rosenbrock, 2>(new Rosenbrock()), x);

	SolverResults results;
	solver.solve(f, method, &results);

	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE ||
	            results.exit_condition == SolverResults::FUNCTION_TOLERANCE ||
	            results.exit_condition == SolverResults::GRADIENT_TOLERANCE);
	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(f.evaluate()), 1e-9);
}

TEST(Solver, NEWTON)
{
	Solver solver;
	solver.log_function = 0;
	test_method(Solver::NEWTON, solver);
}

TEST(Solver, LBFGS)
{
	Solver solver;
	solver.log_function = 0;
	test_method(Solver::LBFGS, solver);
}

TEST(Solver, NELDER_MEAD)
{
	Solver solver;
	solver.log_function = 0;
	solver.maximum_iterations = 10000;
	solver.area_tolerance = 1e-40;
	test_method(Solver::NELDER_MEAD, solver);
}

TEST(Solver, PATTERN_SEARCH)
{
	Solver solver;
	solver.log_function = 0;
	solver.maximum_iterations = 100000;
	test_method(Solver::PATTERN_SEARCH, solver);
}

TEST(Solver, function_tolerance)
{
	Function f;
	double x[2] = {-1.2, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Rosenbrock, 2>(new Rosenbrock()), x);

	Solver solver;
	solver.log_function = 0;
	solver.maximum_iterations = 50;
	solver.gradient_tolerance = 0;
	solver.argument_improvement_tolerance = 0;
	SolverResults results;
	solver.solve_newton(f, &results);

	EXPECT_TRUE(results.exit_condition == SolverResults::FUNCTION_TOLERANCE);
}

TEST(Solver, argument_improvement_tolerance)
{
	Function f;
	double x[2] = {-1.2, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Rosenbrock, 2>(new Rosenbrock()), x);

	Solver solver;
	solver.log_function = 0;
	solver.maximum_iterations = 50;
	solver.gradient_tolerance = 0;
	solver.function_improvement_tolerance = 0;
	SolverResults results;
	solver.solve_newton(f, &results);

	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE);
}

TEST(Solver, gradient_tolerance)
{
	Function f;
	double x[2] = {-1.2, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Rosenbrock, 2>(new Rosenbrock()), x);

	Solver solver;
	solver.log_function = 0;
	solver.maximum_iterations = 50;
	solver.function_improvement_tolerance = 0;
	solver.argument_improvement_tolerance = 0;
	SolverResults results;
	solver.solve_newton(f, &results);

	EXPECT_TRUE(results.exit_condition == SolverResults::GRADIENT_TOLERANCE);
}

struct NanFunctor
{
	template<typename R>
	R operator()(const R* const x) const
	{
		return std::numeric_limits<double>::quiet_NaN();
	}
};

struct InfFunctor
{
	template<typename R>
	R operator()(const R* const x) const
	{
		return std::numeric_limits<double>::infinity();
	}
};

TEST(Solver, inf_nan)
{
	Function f_nan, f_inf;
	double x[1] = {-1.2};
	f_nan.add_variable(x, 1);
	f_inf.add_variable(x, 1);
	f_nan.add_term(new AutoDiffTerm<NanFunctor, 1>(new NanFunctor()), x);
	f_inf.add_term(new AutoDiffTerm<InfFunctor, 1>(new InfFunctor()), x);

	Solver solver;
	solver.log_function = 0;
	SolverResults results;

	solver.solve_newton(f_nan, &results);
	EXPECT_EQ(results.exit_condition, SolverResults::FUNCTION_NAN);

	solver.solve_newton(f_inf, &results);
	EXPECT_EQ(results.exit_condition, SolverResults::FUNCTION_INFINITY);
}

TEST(Solver, L_GBFS_exact)
{
	// Test that the L-BFGS solver follows a reference
	// MATLAB implementation (minFunc) configured to
	// use the same history size and line search method.

	std::vector<int>    iters;
	std::vector<double> fvals;
	iters.push_back(1); fvals.push_back(2.19820e+01);
	iters.push_back(2); fvals.push_back(4.96361e+00);
	iters.push_back(3); fvals.push_back(4.16118e+00);
	iters.push_back(4); fvals.push_back(4.09480e+00);
	iters.push_back(5); fvals.push_back(4.09165e+00);
	iters.push_back(6); fvals.push_back(3.87014e+00);
	iters.push_back(7); fvals.push_back(3.72822e+00);
	iters.push_back(8); fvals.push_back(3.45143e+00);
	iters.push_back(9); fvals.push_back(2.93307e+00);
	iters.push_back(10); fvals.push_back(2.45070e+00);
	iters.push_back(11); fvals.push_back(2.28498e+00);
	iters.push_back(12); fvals.push_back(1.96226e+00);
	iters.push_back(13); fvals.push_back(1.52784e+00);
	iters.push_back(14); fvals.push_back(1.33065e+00);
	iters.push_back(15); fvals.push_back(1.17817e+00);
	iters.push_back(16); fvals.push_back(9.82334e-01);
	iters.push_back(17); fvals.push_back(7.82560e-01);
	iters.push_back(18); fvals.push_back(6.26596e-01);
	iters.push_back(19); fvals.push_back(5.56740e-01);
	iters.push_back(20); fvals.push_back(4.76314e-01);
	iters.push_back(21); fvals.push_back(3.21285e-01);
	iters.push_back(22); fvals.push_back(2.91320e-01);
	iters.push_back(23); fvals.push_back(2.24196e-01);
	iters.push_back(24); fvals.push_back(1.72268e-01);
	iters.push_back(25); fvals.push_back(1.29991e-01);
	iters.push_back(26); fvals.push_back(9.11752e-02);
	iters.push_back(27); fvals.push_back(5.74927e-02);
	iters.push_back(28); fvals.push_back(3.14319e-02);
	iters.push_back(29); fvals.push_back(1.49973e-02);
	iters.push_back(30); fvals.push_back(9.20225e-03);
	iters.push_back(31); fvals.push_back(2.61646e-03);
	iters.push_back(32); fvals.push_back(6.34734e-04);
	iters.push_back(33); fvals.push_back(9.00566e-05);
	iters.push_back(34); fvals.push_back(7.38860e-06);
	iters.push_back(35); fvals.push_back(2.55965e-07);
	iters.push_back(36); fvals.push_back(3.40434e-10);

	for (int i = 0; i < iters.size(); ++i) {
		double x[2] = {-1.2, 1.0};
		Function f;
		f.add_variable(x, 2);
		f.add_term(new AutoDiffTerm<Rosenbrock, 2>(new Rosenbrock()), x);

		Solver solver;
		SolverResults results;
		solver.log_function = 0;
		solver.lbfgs_history_size = 10;
		solver.maximum_iterations = iters[i];

		solver.solve_lbfgs(f, &results);
		double fval = f.evaluate();
		EXPECT_LE( std::abs(fval - fvals[i]) / std::abs(fval), 1e-4);
	}
}

TEST(Solver, Newton_exact)
{
	// Test that the Newton solver follows a reference
	// MATLAB implementation (minFunc) configured to
	// use the same line search method.

	// The Rosenbrock function has a positive definite
	// Hessian at eery iteration step, so this test does
	// not test the iterative Cholesky factorization.

	std::vector<int>    iters;
	std::vector<double> fvals;
	iters.push_back(1); fvals.push_back(4.73188e+000);
	iters.push_back(2); fvals.push_back(4.08740e+000);
	iters.push_back(3); fvals.push_back(3.22867e+000);
	iters.push_back(4); fvals.push_back(3.21390e+000);
	iters.push_back(5); fvals.push_back(1.94259e+000);
	iters.push_back(6); fvals.push_back(1.60019e+000);
	iters.push_back(7); fvals.push_back(1.17839e+000);
	iters.push_back(8); fvals.push_back(9.22412e-001);
	iters.push_back(9); fvals.push_back(5.97489e-001);
	iters.push_back(10); fvals.push_back(4.52625e-001);
	iters.push_back(11); fvals.push_back(2.80762e-001);
	iters.push_back(12); fvals.push_back(2.11393e-001);
	iters.push_back(13); fvals.push_back(8.90195e-002);
	iters.push_back(14); fvals.push_back(5.15354e-002);
	iters.push_back(15); fvals.push_back(1.99928e-002);
	iters.push_back(16); fvals.push_back(7.16924e-003);
	iters.push_back(17); fvals.push_back(1.06961e-003);
	iters.push_back(18); fvals.push_back(7.77685e-005);
	iters.push_back(19); fvals.push_back(2.82467e-007);
	iters.push_back(20); fvals.push_back(8.51707e-012);

	for (int i = 0; i < iters.size(); ++i) {
		double x[2] = {-1.2, 1.0};
		Function f;
		f.add_variable(x, 2);
		f.add_term(new AutoDiffTerm<Rosenbrock, 2>(new Rosenbrock()), x);

		Solver solver;
		SolverResults results;
		solver.log_function = 0;
		solver.maximum_iterations = iters[i];

		solver.solve(f, Solver::NEWTON, &results);
		double fval = f.evaluate();
		EXPECT_LE( std::abs(fval - fvals[i]) / std::abs(fval), 1e-5);
	}
}

struct Quadratic2
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R d0 = x[0] - 2.0;
		R d1 = x[1] + 7.0;
		return 2 * d0*d0 + d1*d1;
	}
};

struct Quadratic2Changed
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R d0 = exp(x[0]) - 2.0;
		R d1 = exp(x[1]) + 7.0;
		return 2 * d0*d0 + d1*d1;
	}
};


//
//	x_i = exp(t_i)
//  t_i = log(x_i)
//
template<int dimension>
class ExpTransform
{
public:
	template<typename R>
	void t_to_x(R* x, const R* t) const
	{
		using std::exp;

		for (size_t i = 0; i < dimension; ++i) {
			x[i] = exp(t[i]);
		}
	}

	template<typename R>
	void x_to_t(R* t, const R* x) const
	{
		using std::log;

		for (size_t i = 0; i < dimension; ++i) {
			t[i] = log(x[i]);
		}
	}

	int x_dimension() const
	{
		return dimension;
	}

	int t_dimension() const
	{
		return dimension;
	}
};

TEST(Solver, SimpleConstraints)
{
	double x[2] = {1, 1};
	Function function;
	function.add_variable(x, 2, new ExpTransform<2>);
	function.add_term(
		new AutoDiffTerm<Quadratic2, 2>(new Quadratic2), x);

	double t[2] = {0, 0};
	Function function_changed;
	function_changed.add_variable(t, 2);
	function_changed.add_term(
		new AutoDiffTerm<Quadratic2Changed, 2>(new Quadratic2Changed), t);

	Solver solver;
	solver.log_function = 0;
	SolverResults results;
	results.exit_condition = SolverResults::NA;

	int max_iter = 1;
	while (! results.exit_success()) {
		solver.maximum_iterations = max_iter++;

		x[0] = x[1] = 1.0;
		t[0] = std::log(x[0]);
		t[1] = std::log(x[1]);
		solver.solve_nelder_mead(function, &results);
		solver.solve_nelder_mead(function_changed, &results);
		EXPECT_NEAR(function.evaluate(), function_changed.evaluate(), 1e-12);

		x[0] = x[1] = 1.0;
		t[0] = std::log(x[0]);
		t[1] = std::log(x[1]);
		solver.solve_lbfgs(function, &results);
		solver.solve_lbfgs(function_changed, &results);
		EXPECT_NEAR(function.evaluate(), function_changed.evaluate(), 1e-12);
	}

	EXPECT_LT( std::abs(x[0] - 2.0), 1e-6);
	EXPECT_LT( std::abs(x[1]), 1e-6);
}

TEST(Solver, PositiveConstraint)
{
	double x[2] = {1, 1};
	Function function;
	function.add_variable(x, 2, new GreaterThanZero(2));
	function.add_term(
		new AutoDiffTerm<Quadratic2, 2>(new Quadratic2), x);

	Solver solver;
	solver.log_function = 0;
	SolverResults results;
	solver.solve_lbfgs(function, &results);

	EXPECT_NEAR(x[0], 2.0, 1e-7);
	EXPECT_NEAR(x[1], 0.0, 1e-7);
}

TEST(Solver, BoxConstraint)
{
	double x[2] = {1, 1};
	Function function;
	double a[2] = {0.0, -0.5};
	double b[2] = {6.0, 10.0};
	function.add_variable(x, 2, new Box(2, a, b));
	function.add_term(
		new AutoDiffTerm<Quadratic2, 2>(new Quadratic2), x);

	Solver solver;
	solver.log_function = 0;
	SolverResults results;
	solver.solve_lbfgs(function, &results);

	EXPECT_NEAR(x[0],  2.0, 1e-4);
	EXPECT_NEAR(x[1], -0.5, 1e-4);
}
