// Petter Strandmark 2013.
//
// The code in this file will be incorporated in the main API
// when more compilers support C++14 generic lambdas.
//

#include <cmath>
#include <limits>
#include <random>

// Check if we’re using November 2013 CTP (or newer).
#ifdef _MSC_FULL_VER
	#if _MSC_FULL_VER > 180021005 // 180021005 is RTM, I believe.
		#define USE_GENERIC_LAMBDAS
	#endif
#endif

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#ifdef USE_GENERIC_LAMBDAS

#include <spii/auto_diff_term.h>
#include <spii/solver.h>
#include <spii/transformations.h>
using namespace spii;

// Creates a differentiable term from a generic lambda and
// argument sizes.
//
// Examples: 
//
//	auto lambda_a =
//		[](auto x)
//		{
//			auto d0 =  x[1] - x[0]*x[0];
//			auto d1 =  1 - x[0];
//			return 100 * d0*d0 + d1*d1;
//		};
//
//	auto term_a = make_term<2>(lambda_a);
//
//
//	auto lambda_b =
//		[](auto x, auto y)
//		{
//			auto d0 =  y[0] - x[0]*x[0];
//			auto d1 =  1 - x[0];
//			return 100 * d0*d0 + d1*d1;
//		};
//
//	auto term_b = make_term<1, 1>(lambda);
// 
template<int... arg_sizes, typename GenericLambda>
std::shared_ptr<Term> make_term(const GenericLambda& lambda)
{
	typedef spii::AutoDiffTerm<GenericLambda, arg_sizes...> TermType;
	return std::make_shared<TermType>(lambda);
}

TEST_CASE("make_term_2")
{
	Function function;
	std::vector<double> x = {0, 0};
	
	auto lambda =
		[](auto x)
		{
			auto d0 =  x[1] - x[0]*x[0];
			auto d1 =  1 - x[0];
			return 100 * d0*d0 + d1*d1;
		};

	auto term = make_term<2>(lambda);

	function.add_term(term, x.data());

	NewtonSolver solver;
	std::stringstream sout;
	solver.log_function =
		[&sout](auto str)
		{
			sout << str << std::endl;
		};

	SolverResults results;
	solver.solve(function, &results);
	INFO(sout.str());
	CHECK(std::abs(x[0] - 1.0) < 1e-8);
	CHECK(std::abs(x[1] - 1.0) < 1e-8);
	CHECK(results.exit_success());
}

TEST_CASE("make_term_1_1")
{
	Function function;
	double x;
	double y;
	
	auto lambda =
		[](auto x, auto y)
		{
			auto d0 =  y[0] - x[0]*x[0];
			auto d1 =  1 - x[0];
			return 100 * d0*d0 + d1*d1;
		};

	auto term = make_term<1, 1>(lambda);

	function.add_term(term, &x, &y);

	NewtonSolver solver;
	std::stringstream sout;
	solver.log_function =
		[&sout](auto str)
		{
			sout << str << std::endl;
		};

	SolverResults results;
	solver.solve(function, &results);
	INFO(sout.str());
	CHECK(std::abs(x - 1.0) < 1e-8);
	CHECK(std::abs(y - 1.0) < 1e-8);
	CHECK(results.exit_success());
}

#endif  // USE_GENERIC_LAMBDAS.
