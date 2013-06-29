// Petter Strandmark 2012.

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <spii/spii.h>

using namespace spii;
using namespace std;

TEST_CASE("spii_assert", "")
{
	CHECK_THROWS_AS(spii_assert(1 == 2), runtime_error);
	spii_assert(1 == 1);
}

TEST_CASE("spii_dassert", "")
{
	#ifdef NDEBUG
		spii_dassert(1 == 2);  // no-op.
	#else
		CHECK_THROWS_AS(spii_dassert(1 == 2), runtime_error);
	#endif
}

TEST_CASE("check", "")
{
	CHECK_THROWS_AS(check(1 == 2, "1 is not 2"), invalid_argument);
	check(1 == 1, "Something is very wrong.");
}
