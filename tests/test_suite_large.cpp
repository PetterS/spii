
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

void run_test_main(const std::function<void(std::vector<double>&, Function*)>& create_f,
                   const std::function<std::vector<double>(int)>& start, 
                   int n,
                   Solver::Method method)
{
	auto this_start = start(n);
	Function f;
	create_f(this_start, &f);
	REQUIRE(f.get_number_of_scalars() == this_start.size());

	std::stringstream information_stream;

	Solver solver;
	solver.log_function =
		[&information_stream](const std::string& str)
		{
			information_stream << str << "\n";
		};
	solver.function_improvement_tolerance = 0;
	solver.argument_improvement_tolerance = 0;
	solver.gradient_tolerance = 1e-7;
	solver.maximum_iterations = 1000000;

	SolverResults results;
	solver.solve(f, method, &results);

	f.print_timing_information(information_stream);
	INFO(information_stream.str());
	INFO(results);

	CHECK(results.exit_success());
}

void run_test(const std::function<void(std::vector<double>&, Function*)>& create_f,
              const std::function<std::vector<double>(int)>& start,
              bool test_newton = true)
{
	if (test_newton) {
		SECTION("Newton-100", "") {
			run_test_main(create_f, start, 100, Solver::NEWTON);
		}
		SECTION("Newton-1000", "") {
			run_test_main(create_f, start, 1000, Solver::NEWTON);
		}
		SECTION("Newton-10000", "") {
			run_test_main(create_f, start, 10000, Solver::NEWTON);
		}
	}

	SECTION("LBFGS-100", "") {
		run_test_main(create_f, start, 100, Solver::LBFGS);
	}
	SECTION("LBFGS-1000", "") {
		run_test_main(create_f, start, 1000, Solver::LBFGS);
	}
	SECTION("LBFGS-10000", "") {
		run_test_main(create_f, start, 10000, Solver::LBFGS);
	}
}

#include "large_suite_andrei.h"
