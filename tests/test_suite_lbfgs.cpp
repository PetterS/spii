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

template<typename Functor, int dimension>
double run_test(double* var, Solver* solver = 0)
{
	Function f;
	f.add_variable(var, dimension);
	f.add_term(new AutoDiffTerm<Functor, dimension>(new Functor()), var);

	Solver own_solver;
	if (solver == 0) {
		solver = &own_solver;
	}
	SolverResults results;
	solver->maximum_iterations = 1000;
	solver->function_improvement_tolerance = 1e-16;
	solver->gradient_tolerance = 1e-12;
	solver->argument_improvement_tolerance = 1e-16;
	solver->lbfgs_history_size = 40;
	solver->solve_lbfgs(f, &results);
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
