
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <spii/google_test_compatibility.h>

#include <spii/auto_diff_term.h>
#include <spii/solver.h>

using namespace spii;

void info_log_function(const std::string& str)
{
	INFO(str);
}

void run_test_main(const std::function<Function(std::vector<double>&)>& function,
                   const std::function<std::vector<double>(int)>& start, 
                   int n,
                   Solver::Method method)
{
	auto this_start = start(n);
	auto f = function(this_start);
	REQUIRE(f.get_number_of_scalars() == this_start.size());

	Solver solver;
	solver.log_function = info_log_function;
	solver.function_improvement_tolerance = 0;
	solver.argument_improvement_tolerance = 0;
	solver.gradient_tolerance = 1e-7;
	solver.maximum_iterations = 1000000;

	SolverResults results;
	solver.solve(f, method, &results);

	CHECK(results.exit_success());
}

void run_test(const std::function<Function(std::vector<double>&)>& function,
              const std::function<std::vector<double>(int)>& start,
              bool test_newton = true)
{
	if (test_newton) {
		SECTION("Newton-100", "") {
			run_test_main(function, start, 100, Solver::NEWTON);
		}
		SECTION("Newton-1000", "") {
			run_test_main(function, start, 1000, Solver::NEWTON);
		}
		SECTION("Newton-10000", "") {
			run_test_main(function, start, 10000, Solver::NEWTON);
		}
	}

	SECTION("LBFGS-100", "") {
		run_test_main(function, start, 100, Solver::LBFGS);
	}
	SECTION("LBFGS-1000", "") {
		run_test_main(function, start, 1000, Solver::LBFGS);
	}
	SECTION("LBFGS-10000", "") {
		run_test_main(function, start, 10000, Solver::LBFGS);
	}
}

#include "large_suite_andrei.h"
