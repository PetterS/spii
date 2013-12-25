// Petter Strandmark 2013.

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <spii/string_utils.h>

using namespace spii;
using namespace std;

TEST_CASE("to_string")
{
	CHECK(to_string("Test",12,"test") == "Test12test");
}
