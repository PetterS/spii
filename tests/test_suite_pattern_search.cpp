// Petter Strandmark 2012
//
#include <cmath>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include <spii/auto_diff_term.h>
#include <spii/solver.h>

using namespace spii;

void create_solver(Solver* solver)
{
	solver->maximum_iterations = 500000;
	solver->area_tolerance = 1e-18;
	solver->function_improvement_tolerance = 1e-12;
}

template<typename Functor, int dimension>
double run_test(double* var, const Solver* solver = 0)
{
	Function f;

	f.add_variable(var, dimension);
	f.add_term(new AutoDiffTerm<Functor, dimension>(new Functor()), var);

	Solver own_solver;
	create_solver(&own_solver);

	if (solver == 0) {
		solver = &own_solver;
	}
	SolverResults results;
	solver->solve_pattern_search(f, &results);
	std::cerr << results;

	for (int i = 0; i < dimension; ++i) {
		std::cout << "x" << i + 1 << " = " << var[i] << ",  ";
	}
	std::cout << std::endl;

	EXPECT_TRUE(results.exit_condition == SolverResults::GRADIENT_TOLERANCE ||
	            results.exit_condition == SolverResults::FUNCTION_TOLERANCE);

	return f.evaluate();
}


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

TEST(Solver, Rosenbrock)
{
	double x[2] = {-1.2, 1.0};
	double fval = run_test<Rosenbrock, 2>(x);

	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(x[1] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(fval), 1e-9);
}

TEST(Solver, RosenbrockFar)
{
	double x[2] = {-1e6, 1e6};

	Solver solver;
	create_solver(&solver);
	solver.gradient_tolerance = 1e-40;
	solver.maximum_iterations = 100000;
	double fval = run_test<Rosenbrock, 2>(x, &solver);

	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(x[1] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(fval), 1e-9);
}

struct Colville
{
	template<typename R>
	R operator()(const R* const x) const
	{
		return 100.0 * (x[1] - x[0]*x[0]) * (x[1] - x[0]*x[0])
		+ (1.0 - x[0]) * (1.0 - x[0])
		+ 90.0 * (x[3] - x[2]*x[2]) * (x[3] - x[2]*x[2])
		+ (1.0 - x[2]) * (1.0 - x[2])
		+ 10.1 * ( (x[1] - 1.0 ) * (x[1] - 1.0)
			+ (x[3] - 1.0) * (x[3] - 1.0) )
		+ 19.8 * (x[1] - 1.0) * (x[3] - 1.0);
	}
};

TEST(Solver, Colville)
{
	double x[4] = {-0.5, 1.0, -0.5, -1.0};
	double fval = run_test<Colville, 4>(x);

	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-8);
	EXPECT_LT( std::fabs(x[1] - 1.0), 1e-8);
	EXPECT_LT( std::fabs(x[2] - 1.0), 1e-8);
	EXPECT_LT( std::fabs(x[3] - 1.0), 1e-8);
}
