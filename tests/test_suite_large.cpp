
#include <cmath>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <spii/solver.h>

// "An analysis of the behavior of a glass of genetic adaptive systems."
// K.A. De Jong.  Ph.D. thesis, University of Michigan, 1975.
struct GeneralizedRosenbrockTerm
{
	template<typename R>
	R operator()(const R* const x1, const R* const x2)
	{
		R d0 =  (*x1) * (*x1) - (*x2);
		R d1 =  1 - (*x1);
		return 100 * d0*d0 + d1*d1;
	}
};

template<size_t n>
void test_rosenbrock()
{
	Function f;
	std::vector<double> x(n);
	for (size_t i = 0; i < n; ++i) {
		f.add_variable(&x[i], 1);
	}

	// Initial values.
	for (size_t i = 0; i < n; ++i) {
		if (i % 2 == 0) {
			x[i] = -1.2;
		}
		else {
			x[i] = 1.0;
		}
	}

	// Add all terms.
	for (size_t i = 0; i < n - 1; ++i) {
		f.add_term(new AutoDiffTerm<GeneralizedRosenbrockTerm, 1, 1>
		               (new GeneralizedRosenbrockTerm()), &x[i], &x[i+1]);
	}

	Solver solver;
	solver.maximum_iterations = 10000;
	SolverResults results;
	solver.Solve(f, &results);

	EXPECT_LT( std::abs(f.evaluate()), 1e-9);

	for (size_t i = 0; i < n - 1; ++i) {
		ASSERT_LT( std::abs(x[i] - 1.0), 1e-9);
	}
}


TEST(Solver, Rosenbrock10)
{
	test_rosenbrock<10>();
}

TEST(Solver, Rosenbrock100)
{
	test_rosenbrock<100>();
}
