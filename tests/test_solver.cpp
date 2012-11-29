
#include <cmath>
#include <limits>
#include <random>

#include <gtest/gtest.h>

#include <spii/auto_diff_term.h>
#include <spii/solver.h>

using namespace spii;

struct Banana
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
	f.add_term(new AutoDiffTerm<Banana, 2>(new Banana()), x);

	Solver solver;
	solver.maximum_iterations = 50;
	SolverResults results;
	solver.Solve(f, &results);

	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE ||
	            results.exit_condition == SolverResults::FUNCTION_TOLERANCE ||
	            results.exit_condition == SolverResults::GRADIENT_TOLERANCE);
	EXPECT_LT( std::abs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::abs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::abs(f.evaluate()), 1e-9);
}

TEST(Solver, function_tolerance)
{
	Function f;
	double x[2] = {-1.2, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Banana, 2>(new Banana()), x);

	Solver solver;
	solver.maximum_iterations = 50;
	solver.gradient_tolerance = 0;
	solver.argument_improvement_tolerance = 0;
	SolverResults results;
	solver.Solve(f, &results);

	EXPECT_TRUE(results.exit_condition == SolverResults::FUNCTION_TOLERANCE);
}

TEST(Solver, argument_improvement_tolerance)
{
	Function f;
	double x[2] = {-1.2, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Banana, 2>(new Banana()), x);

	Solver solver;
	solver.maximum_iterations = 50;
	solver.gradient_tolerance = 0;
	solver.function_improvement_tolerance = 0;
	SolverResults results;
	solver.Solve(f, &results);

	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE);
}

TEST(Solver, gradient_tolerance)
{
	Function f;
	double x[2] = {-1.2, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Banana, 2>(new Banana()), x);

	Solver solver;
	solver.maximum_iterations = 50;
	solver.function_improvement_tolerance = 0;
	solver.argument_improvement_tolerance = 0;
	SolverResults results;
	solver.Solve(f, &results);

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
	SolverResults results;

	solver.Solve(f_nan, &results);
	EXPECT_EQ(results.exit_condition, SolverResults::FUNCTION_NAN);

	solver.Solve(f_inf, &results);
	EXPECT_EQ(results.exit_condition, SolverResults::FUNCTION_INFINITY);
}