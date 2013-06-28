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
	solver->maximum_iterations = 1000;
	solver->function_improvement_tolerance = 1e-16;
	solver->gradient_tolerance = 1e-12;
	solver->argument_improvement_tolerance = 1e-16;
	solver->lbfgs_history_size = 40;

	solver->log_function = info_log_function;
}

template<typename Functor, int dimension>
double run_test(double* var, const Solver* solver = 0)
{
	Function f;
	f.hessian_is_enabled = false;

	f.add_variable(var, dimension);
	f.add_term(std::make_shared<AutoDiffTerm<Functor, dimension>>(), var);

	Solver own_solver;
	create_solver(&own_solver);

	if (solver == 0) {
		solver = &own_solver;
	}
	SolverResults results;
	global_string_stream.str("");
	solver->solve_lbfgs(f, &results);
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

	return f.evaluate();
}

#include "suite_more_et_al.h"
#include "suite_test_opt.h"
#include "suite_uctp.h"

