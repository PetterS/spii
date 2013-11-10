// Petter Strandmark 2013.

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <spii/auto_diff_term.h>
#include <spii/constrained_function.h>
#include <spii/solver.h>

using namespace spii;
using namespace std;

class ObjectiveTerm
{
public:
	template<typename R>
	R operator()(const R* const x) const
	{
		auto dx = x[0] - 2.0;
		auto dy = x[1] - 2.0;
		return  dx*dx + dy*dy;
	}
};

//  x·x + y·y ≤ 1
class ConstraintTerm
{
public:
	template<typename R>
	R operator()(const R* const x) const
	{
		auto dx = x[0];
		auto dy = x[1];
		return  dx*dx + dy*dy - 1;
	}
};

//  x ≤ 100
class LessThan100
{
public:
	template<typename R>
	R operator()(const R* const x) const
	{
		auto dx = x[0];
		return  dx - 100;
	}
};

void test_simple_constrained_function(const std::vector<double> x_start, bool feasible)
{
	ConstrainedFunction function;
	stringstream log_stream;
	LBFGSSolver solver;
	solver.log_function =
		[&log_stream](const string& str)
		{
			log_stream << str << endl;
		};

	auto x = x_start;

	function.add_term(make_shared<AutoDiffTerm<ObjectiveTerm, 2>>(), &x[0]);
	function.add_constraint_term("circle", make_shared<AutoDiffTerm<ConstraintTerm, 2>>(), &x[0]);
	function.add_constraint_term("less than 100", make_shared<AutoDiffTerm<LessThan100, 2>>(), &x[0]);
	CHECK(function.is_feasible() == feasible);
	
	SolverResults results;
	function.solve(solver, &results);
	log_stream << results << endl;

	INFO(log_stream.str());
	CAPTURE(x[0]);
	CAPTURE(x[1]);
	CHECK((x[0]*x[0] + x[1]*x[1]) <= (1.0 + 1e-8));

	auto dx = x[0] - 2;
	auto dy = x[1] - 2;
	CHECK(abs(function.objective().evaluate() - (dx*dx + dy*dy)) <= 1e-14);
}

TEST_CASE("feasible_start")
{
	test_simple_constrained_function({0.0, 0.0}, true);
}


TEST_CASE("infeasible_start")
{
	test_simple_constrained_function({10.0, 4.0}, false);
}

TEST_CASE("max_iterations")
{
	ConstrainedFunction function;
	function.max_number_of_iterations = 1;
	vector<double> x = {0, 0};
	function.add_term(make_shared<AutoDiffTerm<ObjectiveTerm, 2>>(), &x[0]);
	function.add_constraint_term("circle", make_shared<AutoDiffTerm<ConstraintTerm, 2>>(), &x[0]);
	function.add_constraint_term("less than 100", make_shared<AutoDiffTerm<LessThan100, 2>>(), &x[0]);
	LBFGSSolver solver;
	solver.log_function = nullptr;
	SolverResults results;
	function.solve(solver, &results);
	CHECK(results.exit_condition == SolverResults::NO_CONVERGENCE);
}

TEST_CASE("no_readding_constraints")
{
	ConstrainedFunction function;
	double x[2] = {0, 0};
	function.add_constraint_term("circle", make_shared<AutoDiffTerm<ConstraintTerm, 2>>(), &x[0]);
	CHECK_THROWS(function.add_constraint_term("circle", make_shared<AutoDiffTerm<ConstraintTerm, 2>>(), &x[0]));
}
