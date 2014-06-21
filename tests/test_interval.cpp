// Petter Strandmark 2013.

#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <spii/interval.h>
using namespace spii;

TEST_CASE("Interval/get_set", "Tests get and set functions")
{
	Interval<double> i1(123, 456);

	CHECK(i1.get_lower() == 123);
	CHECK(i1.get_upper() == 456);
}

TEST_CASE("Interval/plus", "")
{
	Interval<double> i1(1, 2);
	Interval<double> i2(3, 4);

	INFO(i1 << " + " << 10 << " = " << i1 + 10.0);
	CHECK((i1 + 10.0) == Interval<double>(11, 12));
	CHECK((10.0 + i1) == Interval<double>(11, 12));
	CHECK((i1 + i2) == Interval<double>(4, 6));
}

TEST_CASE("Interval/minus", "")
{
	Interval<double> i1(1, 2);
	Interval<double> i2(3, 4);

	CHECK((i1 - 10.0) == Interval<double>(-9, -8));
	CHECK((10.0 - i1) == Interval<double>(8, 9));
	CHECK((i1 - i2) == Interval<double>(-3, -1));
	CHECK((i1 - i2) == -(i2 - i1));
}

TEST_CASE("Interval/multiplication", "")
{
	Interval<double> i1(1, 2);

	CHECK((i1 *  5.0) == Interval<double>(5, 10));
	CHECK((5.0  * i1) == Interval<double>(5, 10));
	CHECK((i1 * -5.0) == Interval<double>(-10, -5));
	CHECK((-5.0 * i1) == Interval<double>(-10, -5));

	Interval<double> i2(-1, 2);

	CHECK((i2 *  5.0) == Interval<double>(-5, 10));
	CHECK((5.0  * i2) == Interval<double>(-5, 10));
	CHECK( (i2 * -5.0) == Interval<double>(-10, 5));
	CHECK((-5.0 * i2) == Interval<double>(-10, 5));

	CHECK((i1 * i2) == Interval<double>(-2, 4));
}

TEST_CASE("multiplication_all_sign_combinations")
{
	int lim = 3;
	for (int i1 = -lim; i1 <= +lim; ++i1) {
	for (int i2 = i1 + 1; i2 <= +lim; ++i2) {
		for (int j1 = -lim; j1 <= +lim; ++j1) {
		for (int j2 = j1 +1; j2 <= +lim; ++j2) {
			Interval<int> interval1(i1, i2);
			Interval<int> interval2(j1, j2);
			CHECK((interval1 * interval2) == (interval2 * interval1));
		}}
	}}
}

TEST_CASE("Interval/division", "")
{
	Interval<double> result;

	Interval<double> i1(1.0, 2.0);
	result = i1 / 3.0;
	CHECK(result.get_lower() == 1.0 / 3.0);
	CHECK(result.get_upper() == 2.0 / 3.0);

	Interval<double> i2(-1.0, 2.0);
	result = i2 / 3.0;
	CHECK(result.get_lower() == - 1.0 / 3.0);
	CHECK(result.get_upper() ==   2.0 / 3.0);

	Interval<double> i3(1.0, 2.0);
	result = i3 / -3.0;
	CHECK(result.get_lower() == - 2.0 / 3.0);
	CHECK(result.get_upper() == - 1.0 / 3.0);

	Interval<double> i4(1.0, 2.0);
	result = 3.0 / i4;
	CHECK(result.get_lower() == 3.0 / 2.0);
	CHECK(result.get_upper() == 3.0);

	Interval<double> i5(-1.0, 1.0);
	result = 3.0 / i5;
	CHECK(result.get_lower() == - std::numeric_limits<double>::infinity());
	CHECK(result.get_upper() ==   std::numeric_limits<double>::infinity());
}

TEST_CASE("division_strictly_positive")
{
	int lim = 4;
	for (int i1 = 1; i1 <= +lim; ++i1) {
	for (int i2 = i1 + 1; i2 <= +lim; ++i2) {
		for (int j1 = 1; j1 <= +lim; ++j1) {
		for (int j2 = j1 +1; j2 <= +lim; ++j2) {
			if (i1 != 0 && i2 != 0) {
				Interval<double> interval1(i1, i2);
				Interval<double> interval2(j1, j2);
				CAPTURE(interval1);
				CAPTURE(interval2);
				CHECK((interval1 / interval2).get_lower() > 0);
			}
		}}
	}}
}

TEST_CASE("division_as_multiplication")
{
	int lim = 4;
	for (int i1 = -lim; i1 <= +lim; ++i1) {
	for (int i2 = i1 + 1; i2 <= +lim; ++i2) {
		for (int j1 = 1; j1 <= +lim; ++j1) {
		for (int j2 = j1 +1; j2 <= +lim; ++j2) {
			if (i1 != 0 && i2 != 0) {
				Interval<double> interval1(i1, i2);
				Interval<double> interval2(j1, j2);
				CAPTURE(interval1);
				CAPTURE(interval2);
				auto inv2 = 1.0 / interval2;
				CHECK((interval1 / interval2) == (inv2 * interval1));
			}
		}}
	}}
}

TEST_CASE("Interval/cos", "")
{
	Interval<double> result;

	Interval<double> i1(2.0, 6.0);
	result = cos(i1);
	CHECK(result.get_lower() == -1.0);
	CHECK(Approx(result.get_upper()) == cos(6.0));

	Interval<double> i2(-10.0, -4.0);
	result = cos(i2);
	CHECK(result.get_lower() == -1.0);
	CHECK(result.get_upper() == +1.0);

	Interval<double> i3(-10.0, -8.0);
	result = cos(i3);
	CHECK(result.get_lower() == -1.0);
	CHECK(Approx(result.get_upper()) == cos(8.0));

	Interval<double> i4(4.0, 8.0);
	result = cos(i4);
	CHECK(Approx(result.get_lower()) == cos(4.0));
	CHECK(result.get_upper() == +1.0);

	Interval<double> i5(2.0, 2.5);
	result = cos(i5);
	CHECK(Approx(result.get_lower()) == cos(2.5));
	CHECK(Approx(result.get_upper()) == cos(2.0));
}
