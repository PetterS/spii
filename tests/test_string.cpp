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

TEST_CASE("format_string_1")
{
	CHECK(format_string("Test %0!", 12) == "Test 12!");
	CHECK(format_string("Test %0", 12) == "Test 12");
}

TEST_CASE("format_string_%")
{
	CHECK(format_string("Test %0%%", 12) == "Test 12%");
	CHECK(format_string("Test %0%%!", 12) == "Test 12%!");
	CHECK(format_string("Test %0 %%", 12) == "Test 12 %");
	CHECK(format_string("Test %0 %%!", 12) == "Test 12 %!");
}

TEST_CASE("format_string_2")
{
	CHECK(format_string("Test %0 and %1!", 'A', 'O') == "Test A and O!");
	CHECK(format_string("Test %1 and %0!", 'A', 'O') == "Test O and A!");
}
