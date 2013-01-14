// Petter Strandmark 2012.

#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <cstddef>

#include <spii/auto_diff_term.h>
#include <spii/solver.h>

using namespace spii;

struct LinearObjective
{
	std::vector<double> c;
	LinearObjective( const std::vector<double>& c)
	{
		this->c = c;
	}

	template<typename R>
	R operator()(const R* const x) const
	{
		R f = 0.0;
		for (size_t i = 0; i < c.size(); ++i) {
			f += c[i] * x[i];
		}
		return f;
	}
};

// Barrier for aTx <= b.
struct LogBarrier
{
	std::vector<double> a;
	double b;
	double* mu;
	LogBarrier(const std::vector<double>& a, double b, double* mu)
	{
		this->a = a;
		this->b = b;
		this->mu = mu;
	}

	template<typename R>
	R operator()(const R* const x) const
	{
		R constraint = -b;
		for (size_t i = 0; i < a.size(); ++i) {
			constraint += a[i] * x[i];
		}

		// constraint <= 0 is converted to a barrier
		// - mu * log(-constraint).
		return - (*mu) * log(-constraint);
	}
};

int main()
{
	std::mt19937 prng(0);
	std::normal_distribution<double> normal;
	auto randn = std::bind(normal, prng);

	#define n 10

	// Variables.
	std::vector<double> x(n, 0.0);

	// Barrier parameter.
	double mu = 1.0;

	// Objective function.
	Function f;
	f.add_variable(&x[0], n);

	// Generate random objective vector.
	std::vector<double> c(n);
	for (size_t i = 0; i < n; ++i) {
		c[i] = randn();
	}
	f.add_term(new AutoDiffTerm<LinearObjective, n>(
		new LinearObjective(c)), &x[0]);

	// sum x_i <= 10
	double b = 10;
	std::vector<double> a1(n, 1.0);
	f.add_term(new AutoDiffTerm<LogBarrier, n>(
		new LogBarrier(a1, b, &mu)), &x[0]);

	//  sum x_i >=  10  <=>
	// -sum x_i <= -10
	std::vector<double> a2(n, -1.0);
	b = 10;
	f.add_term(new AutoDiffTerm<LogBarrier, n>(
		new LogBarrier(a2, b, &mu)), &x[0]);

	// Add barriers for individual scalars.
	//
	//		-100 <= x[i] <= 100
	//
	for (size_t i = 0; i < n; ++i) {
		std::vector<double> a3(n, 0.0);
		std::vector<double> a4(n, 0.0);
		b = 100;
		a3[i] =  1.0;
		a4[i] = -1.0;
		f.add_term(new AutoDiffTerm<LogBarrier, n>(
			new LogBarrier(a3, b, &mu)), &x[0]);
		f.add_term(new AutoDiffTerm<LogBarrier, n>(
			new LogBarrier(a4, b, &mu)), &x[0]);
	}

	Solver solver;
	solver.sparsity_mode = Solver::DENSE;
	solver.maximum_iterations = 100;
	// nullptr does not work in gcc 4.5
	solver.log_function = [](const std::string&) { };
	SolverResults results;

	for (int iter = 1; iter <= 8; ++iter) {
		solver.solve_newton(f, &results);

		double sumx = 0.0;
		double cTx  = 0.0;
		double minx = std::numeric_limits<double>::infinity();
		double maxx = -minx;
		for (size_t i = 0; i < n; ++i) {
			cTx  += x[i] * c[i];
			sumx += x[i];
			minx = std::min(x[i], minx);
			maxx = std::max(x[i], maxx);
		}

		std::printf("mu=%.2e, f=%+.4e, cTx=%+.4e, sum(x)=%+.4e, (min,max)(x)=(%.3e, %3e)\n",
			mu, f.evaluate(), cTx, sumx, minx, maxx);

		mu /= 10.0;
	}
	std::cerr << "Solution to the linear programming problem is cTx = " << f.evaluate() << '\n';
}
