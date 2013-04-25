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
#include <sstream>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <spii/google_test_compatibility.h>

#include <spii/auto_diff_term.h>
#include <spii/solver.h>

using namespace spii;

std::stringstream global_string_stream;
void info_log_function(const std::string& str)
{
	global_string_stream << str << "\n";
}

void create_solver(Solver* solver)
{
	// Use default solver settings.

	solver->log_function = info_log_function;
}

int cumulative_iterations   = 0;
int cumulative_evalutations = 0;
double cumulative_time      = 0;

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
	global_string_stream.str("");
	solver->solve_newton(f, &results);
	INFO(global_string_stream.str());
	INFO(results);

	std::stringstream sout;
	for (int i = 0; i < dimension; ++i) {
		sout << "x" << i + 1 << " = " << var[i] << ",  ";
	}
	INFO(sout.str());

	EXPECT_TRUE(results.exit_condition == SolverResults::ARGUMENT_TOLERANCE ||
	            results.exit_condition == SolverResults::FUNCTION_TOLERANCE ||
	            results.exit_condition == SolverResults::GRADIENT_TOLERANCE);

	cumulative_evalutations += f.evaluations_with_gradient;
	cumulative_time         += results.total_time - results.log_time;
	INFO("Cumulative evaluations: " << cumulative_evalutations);
	INFO("Cumulative time       : " << cumulative_time);

	return f.evaluate();
}

#include "test_suite_include.h"
