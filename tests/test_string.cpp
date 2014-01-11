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

TEST_CASE("to_string_pair")
{
	pair<int, string> p{123, "Test"};
	CHECK(to_string(p) == "(123, Test)");
}

TEST_CASE("to_string_pair_pair")
{
	auto p  = make_pair(123, "Test");
	auto pp = make_pair(12.1, p);
	CHECK(to_string(pp) == "(12.1, (123, Test))");
}

TEST_CASE("to_string_vector")
{
	vector<int> v{1, 2, 3};
	CHECK(to_string(v) == "[1, 2, 3]");
	v.clear();
	CHECK(to_string(v) == "[]");
}

TEST_CASE("to_string_vector_pair")
{
	vector<pair<int, string>> v{{1, "P"}, {2, "S"}};
	CHECK(to_string(v) == "[(1, P), (2, S)]");
}

TEST_CASE("to_string_set")
{
	set<int> v{1, 2, 3};
	CHECK(to_string(v) == "{1, 2, 3}");
	v.clear();
	CHECK(to_string(v) == "{}");
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

TEST_CASE("from_string")
{
	CHECK(from_string<int>("42") == 42);
	CHECK(from_string("asd", 42) == 42);
	CHECK_THROWS(from_string<int>("abc"));
}
