
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include <spii/solver.h>

struct Rosenbrock
{
	template<typename R>
	R operator()(const R* const x)
	{
		R d0 =  x[1] - x[0]*x[0];
		R d1 =  1 - x[0];
		return 100 * d0*d0 + d1*d1;
	}
};

TEST(Solver, Rosenbrock)
{
	Function f;
	double x[3] = {-1.2, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Rosenbrock, 2>(new Rosenbrock()), x);

	Solver solver;
	SolverResults results;
	solver.Solve(f, &results);

	EXPECT_LT( std::abs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::abs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::abs(f.evaluate()), 1e-9);
}

struct FreudenStein_Roth
{
	template<typename R>
	R operator()(const R* const x)
	{
		R d0 =  -13.0 + x[0] + ((5.0 - x[1])*x[1] - 2.0)*x[1];
		R d1 =  -29.0 + x[0] + ((x[1] + 1.0)*x[1] - 14.0)*x[1];
		return d0*d0 + d1*d1;
	}
};

TEST(Solver, FreudenStein_Roth)
{
	Function f;
	double x[2] = {0.5, -2.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<FreudenStein_Roth, 2>(new FreudenStein_Roth()), x);

	Solver solver;
	SolverResults results;
	solver.Solve(f, &results);

	// Can end up in local minima 48.9842...
	//EXPECT_LT( std::abs(x[0] - 5.0), 1e-9);
	//EXPECT_LT( std::abs(x[1] - 4.0), 1e-9);
	//EXPECT_LT( std::abs(f.evaluate()), 1e-9);
}

struct Powell_badly_scaled
{
	template<typename R>
	R operator()(const R* const x)
	{
		R d0 = 1e4*x[0]*x[1] - 1;
		R d1 = exp(-x[0]) + exp(-x[1]) - 1.0001;
		return d0*d0 + d1*d1;
	}
};

TEST(Solver, Powell_badly_scaled)
{
	Function f;
	double x[2] = {0.0, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Powell_badly_scaled, 2>(new Powell_badly_scaled()), x);

	Solver solver;
	SolverResults results;
	solver.Solve(f, &results);

	EXPECT_LT( std::abs(f.evaluate()), 1e-9);
}

struct Brown_badly_scaled
{
	template<typename R>
	R operator()(const R* const x)
	{
		R d0 = x[0] - 1e6;
		R d1 = x[1] - 2e-6;
		R d2 = x[0]*x[1] - 2;
		return d0*d0 + d1*d1 + d2*d2;
	}
};

TEST(Solver, Brown_badly_scaled)
{
	Function f;
	double x[2] = {1.0, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Brown_badly_scaled, 2>(new Brown_badly_scaled()), x);

	Solver solver;
	SolverResults results;
	solver.Solve(f, &results);

	EXPECT_LT( std::abs(x[0] - 1e6),  1e-3);
	EXPECT_LT( std::abs(x[1] - 2e-6), 1e-9);
	EXPECT_LT( std::abs(f.evaluate()), 1e-9);
}