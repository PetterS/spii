// Petter Strandmark 2012
//
// Test functions from
// Jorge J. More, Burton S. Garbow and Kenneth E. Hillstrom,
// "Testing unconstrained optimization software",
// Transactions on Mathematical Software 7(1):17-41, 1981.
// http://www.caam.rice.edu/~zhang/caam454/nls/MGH.pdf
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
	// Use default solver settings.
}

template<typename Functor, int dimension>
double run_test(double* var, const Solver* solver = 0)
{
	Function f;
	f.add_variable(var, dimension);
	f.add_term(new AutoDiffTerm<Functor, dimension>(new Functor()), var);

	Solver own_solver;
	if (solver == 0) {
		solver = &own_solver;
	}
	SolverResults results;
	solver->solve_newton(f, &results);
	std::cerr << results;

	for (int i = 0; i < dimension; ++i) {
		std::cout << "x" << i + 1 << " = " << var[i] << ",  ";
	}
	std::cout << std::endl;

	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE ||
	            results.exit_condition == SolverResults::FUNCTION_TOLERANCE ||
	            results.exit_condition == SolverResults::GRADIENT_TOLERANCE);

	return f.evaluate();
}

#include "test_suite_include.h"
