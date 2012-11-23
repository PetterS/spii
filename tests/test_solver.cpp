
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include <spii/solver.h>

struct Banana
{
	template<typename R>
	R operator()(const R* const x)
	{
		R d0 =  x[1] - x[0]*x[0];
		R d1 =  1 - x[0];
		return 100 * d0*d0 + d1*d1;
	}
};

TEST(Solver, banana)
{
	Function f;
	double x[3] = {-1.2, 1.0};
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
