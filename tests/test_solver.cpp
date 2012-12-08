
#include <cmath>
#include <limits>
#include <random>

#include <gtest/gtest.h>

#include <spii/auto_diff_term.h>
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

TEST(Solver, banana)
{
	Function f;
	double x[2] = {-1.2, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Rosenbrock, 2>(new Rosenbrock()), x);

	Solver solver;
	solver.log_function = 0;
	solver.maximum_iterations = 50;
	SolverResults results;
	solver.solve_newton(f, &results);

	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE ||
	            results.exit_condition == SolverResults::FUNCTION_TOLERANCE ||
	            results.exit_condition == SolverResults::GRADIENT_TOLERANCE);
	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(f.evaluate()), 1e-9);
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

TEST(Solver, L_GBFS)
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

TEST(Solver, NelderMead)
{
	Function f;
	double x[2] = {-1.2, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Rosenbrock, 2>(new Rosenbrock()), x);

	Solver solver;
	solver.maximum_iterations = 500;
	solver.log_function = 0;
	SolverResults results;
	solver.solve_nelder_mead(f, &results);
	solver.solve_newton(f, &results);

	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE ||
	            results.exit_condition == SolverResults::FUNCTION_TOLERANCE ||
	            results.exit_condition == SolverResults::GRADIENT_TOLERANCE);
	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(f.evaluate()), 1e-9);
}