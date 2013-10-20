// Petter Strandmark 2012--2013
//
// Test functions from
// Jorge J. More, Burton S. Garbow and Kenneth E. Hillstrom,
// "Testing unconstrained optimization software",
// Transactions on Mathematical Software 7(1):17-41, 1981.
// http://www.caam.rice.edu/~zhang/caam454/nls/MGH.pdf
//
#include <cmath>
#include <iostream>
#include <memory>
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

std::unique_ptr<Solver> create_solver()
{
	std::unique_ptr<Solver> solver(new NewtonSolver);
	// Use default solver settings.

	solver->log_function = info_log_function;
	//solver->factorization_method = Solver::ITERATIVE;

	return std::move(solver);
}

int cumulative_iterations   = 0;
int cumulative_evalutations = 0;
double cumulative_time      = 0;

template<typename Functor, int dimension>
double run_test(double* var, const Solver* solver = 0)
{
	Function f;
	f.add_variable(var, dimension);
	f.add_term(std::make_shared<AutoDiffTerm<Functor, dimension>>(), var);

	auto own_solver = create_solver();
	if (solver == 0) {
		solver = own_solver.get();
	}
	SolverResults results;
	global_string_stream.str("");
	solver->solve(f, &results);
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

#include "suite_more_et_al.h"
#include "suite_test_opt.h"
#include "suite_uctp.h"
