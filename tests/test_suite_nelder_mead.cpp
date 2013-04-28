// Petter Strandmark 2012
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
	// The problems in the test suite are rather difficult
	// and often require very exact solutions (for N-M).
	// Therefore, many iterations have to be used.
	solver->maximum_iterations = 100000;
	solver->area_tolerance   = 0;  // Not used.
	solver->length_tolerance = 1e-20;

	solver->log_function = info_log_function;
}

template<typename Functor, int dimension>
double run_test(double* var, const Solver* solver = 0)
{
	Function f;

	f.add_variable(var, dimension);
	f.add_term(new AutoDiffTerm<Functor, dimension>(new Functor()), var);

	// Compute and save the gradient.
	Eigen::VectorXd xvec(dimension);
	for (size_t i = 0; i < dimension; ++i) {
		xvec[i] = var[i];
	}
	Eigen::VectorXd gradient(dimension);
	f.evaluate(xvec, &gradient);
	double normg0 = std::max(gradient.maxCoeff(), -gradient.minCoeff());

	Solver own_solver;
	create_solver(&own_solver);

	if (solver == 0) {
		solver = &own_solver;
	}
	SolverResults results;
	global_string_stream.str("");
	solver->solve_nelder_mead(f, &results);
	INFO(global_string_stream.str());
	INFO(results);

	std::stringstream sout;
	for (int i = 0; i < dimension; ++i) {
		sout << "x" << i + 1 << " = " << var[i] << ",  ";
	}
	INFO(sout.str());

	EXPECT_TRUE(results.exit_condition == SolverResults::GRADIENT_TOLERANCE);

	// Check that the gradient actually vanishes.
	// This test fails for one test, even though the correct
	// objective function value is reached.
	for (size_t i = 0; i < dimension; ++i) {
		xvec[i] = var[i];
	}
	double fval = f.evaluate(xvec, &gradient);
	double normg = std::max(gradient.maxCoeff(), -gradient.minCoeff());
	INFO("||g0|| = " << normg0);
	INFO("||g||  = " << normg);
	INFO(typeid(Functor).name());

	std::string problem_name = typeid(Functor).name();
	// The "Easom" problem does not return a very reliable gradient at the
	// optimum. The test makes sure that the optimal point is found.
	if (problem_name.find("Easom") == std::string::npos) {
		EXPECT_LT(normg / normg0, 1e-5);
	}

	return fval;
}

#include "suite_more_et_al.h"
#include "suite_test_opt.h"
#include "suite_uctp.h"

