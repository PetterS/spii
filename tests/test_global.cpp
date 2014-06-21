// Petter Strandmark 2013.

#include <queue>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <spii/interval_term.h>
#include <spii/solver.h>
using namespace spii;

struct SimpleFunction1
{
	template<typename R>
	R operator()(const R* const x) const
	{
		return pow(x[0], 2) + 3.0*cos(5.0*x[0]);
	}
};

struct SimpleFunction2
{
	template<typename R>
	R operator()(const R* const x) const
	{
		return x[0]*x[0] + x[1]*x[1] + 1.0;
	}
};

struct SimpleFunction1_1
{
	template<typename R>
	R operator()(const R* const x, const R* const y) const
	{
		return x[0]*x[0] + y[0]*y[0] + 1.0;
	}
};

TEST_CASE("global_optimization/simple_function1", "Petter")
{
	double x = 2.0;
	Function f;
	f.add_variable(&x, 1);
	f.add_term(std::make_shared<IntervalTerm<SimpleFunction1, 1>>(),
	           &x);

	GlobalSolver solver;
	solver.maximum_iterations = 1015;
	std::stringstream info_buffer;
	solver.log_function = [&info_buffer](const std::string& str) { info_buffer << str << std::endl; };
	SolverResults results;
	std::vector<Interval<double>> x_interval;
	x_interval.push_back(Interval<double>(-10.0, 9.0));
	auto interval = solver.solve_global(f, x_interval, &results);
	REQUIRE(interval.size() == 1);

	INFO(info_buffer.str());
	INFO(results);

	auto opt = Interval<double>(results.optimum_lower, results.optimum_upper);
	CHECK((opt.get_upper() - opt.get_lower()) <= 1e-3);
	auto val = f.evaluate();
	CHECK(opt.get_lower() <= val); CHECK(val <= opt.get_upper());
}

TEST_CASE("global_optimization/simple_function2", "Petter")
{
	double x[] = {2.0, 2.0};
	Function f;
	f.add_variable(x, 2);
	f.add_term(std::make_shared<IntervalTerm<SimpleFunction2, 2>>(),
	           x);

	GlobalSolver solver;
	solver.maximum_iterations = 1000;
	solver.argument_improvement_tolerance = 0;
	solver.function_improvement_tolerance = 1e-12;
	std::stringstream info_buffer;
	solver.log_function = [&info_buffer](const std::string& str) { info_buffer << str << std::endl; };
	SolverResults results;
	std::vector<Interval<double>> x_interval;
	x_interval.push_back(Interval<double>(-10.0, 9.0));
	x_interval.push_back(Interval<double>(-8.0, 8.0));
	auto interval = solver.solve_global(f, x_interval, &results);
	REQUIRE(interval.size() == 2);

	INFO(info_buffer.str());
	INFO(results);

	auto opt = Interval<double>(results.optimum_lower, results.optimum_upper);
	CHECK((opt.get_upper() - opt.get_lower()) <= 1e-10);
	CHECK(results.exit_condition == SolverResults::FUNCTION_TOLERANCE);
	double ground_truth = 1.0;
	CHECK(opt.get_lower() <= ground_truth); CHECK(ground_truth <= opt.get_upper());
}

TEST_CASE("global_optimization/simple_function1-1", "Petter")
{
	double x = 2.0, y = 2.0;
	Function f;
	f.add_variable(&x, 1);
	f.add_variable(&y, 1);
	f.add_term(std::make_shared<IntervalTerm<SimpleFunction1_1, 1, 1>>(),
	           &x, &y);

	GlobalSolver solver;
	solver.maximum_iterations = 1000;
	solver.argument_improvement_tolerance = 0;
	solver.function_improvement_tolerance = 1e-12;
	std::stringstream info_buffer;
	solver.log_function = [&info_buffer](const std::string& str) { info_buffer << str << std::endl; };
	SolverResults results;
	std::vector<Interval<double>> x_interval;
	x_interval.push_back(Interval<double>(-10.0, 9.0));
	x_interval.push_back(Interval<double>(-8.0, 8.0));
	auto interval = solver.solve_global(f, x_interval, &results);
	REQUIRE(interval.size() == 2);

	INFO(info_buffer.str());
	INFO(results);

	auto opt = Interval<double>(results.optimum_lower, results.optimum_upper);
	CHECK((opt.get_upper() - opt.get_lower()) <= 1e-10);
	CHECK(results.exit_condition == SolverResults::FUNCTION_TOLERANCE);
	double ground_truth = 1.0;
	CHECK(opt.get_lower() <= ground_truth); CHECK(ground_truth <= opt.get_upper());
}

TEST_CASE("Constant variable")
{
	double x = 2.0;
	double y = 2.0;
	Function f;
	f.add_variable(&x, 1);
	f.add_variable(&y, 1);
	f.add_term(std::make_shared<IntervalTerm<SimpleFunction1_1, 1, 1>>(),
	           &x, &y);

	GlobalSolver solver;
	solver.maximum_iterations = 1000;
	solver.argument_improvement_tolerance = 0;
	solver.function_improvement_tolerance = 1e-12;

	{
		f.set_constant(&y, true);
		REQUIRE(f.get_number_of_variables() == 2);
		REQUIRE(f.get_number_of_scalars() == 1);

		std::stringstream info_buffer;
		solver.log_function = [&info_buffer](const std::string& str) { info_buffer << str << std::endl; };
		SolverResults results;
		std::vector<Interval<double>> x_interval;
		x_interval.push_back(Interval<double>(-10.0, 9.0));

		auto interval = solver.solve_global(f, x_interval, &results);
		REQUIRE(interval.size() == 1);
		INFO(info_buffer.str());
		INFO(results);

		auto opt = Interval<double>(results.optimum_lower, results.optimum_upper);
		CHECK((opt.get_upper() - opt.get_lower()) <= 1e-10);
		CHECK(results.exit_condition == SolverResults::FUNCTION_TOLERANCE);
		double ground_truth = 5.0;
		CHECK(opt.get_lower() <= ground_truth); CHECK(ground_truth <= opt.get_upper());
	}

	{
		f.set_constant(&y, false);
		x = 3.0;
		f.set_constant(&x, true);
		REQUIRE(f.get_number_of_variables() == 2);
		REQUIRE(f.get_number_of_scalars() == 1);

		std::stringstream info_buffer;
		solver.log_function = [&info_buffer](const std::string& str) { info_buffer << str << std::endl; };

		SolverResults results;
		std::vector<Interval<double>> x_interval;
		x_interval.push_back(Interval<double>(-10.0, 10.0));
		auto interval = solver.solve_global(f, x_interval, &results);
		REQUIRE(interval.size() == 1);
		INFO(info_buffer.str());
		INFO(results);

		Interval<double> opt(results.optimum_lower, results.optimum_upper);
		CHECK((opt.get_upper() - opt.get_lower()) <= 1e-10);
		CHECK(results.exit_condition == SolverResults::FUNCTION_TOLERANCE);
		double ground_truth = 10.0;
		CHECK(opt.get_lower() <= ground_truth); CHECK(ground_truth <= opt.get_upper());
	}
}
