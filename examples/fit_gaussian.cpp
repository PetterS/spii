// Petter Strandmark 2012--2013.

#include <functional>
#include <iostream>
#include <random>

#include <spii/auto_diff_term.h>
#include <spii/constraints.h>
#include <spii/interval_term.h>
#include <spii/solver.h>

using namespace spii;

// One term in the negative log-likelihood function for
// a one-dimensional Gaussian distribution.
struct NegLogLikelihood
{
	double sample;
	NegLogLikelihood(double sample)
	{
		this->sample = sample;
	}

	template<typename R>
	R operator()(const R* const mu, const R* const sigma) const
	{
		R diff = (*mu - sample) / *sigma;
		return 0.5 * diff*diff + log(*sigma);
	}
};

int main_function()
{
	std::mt19937 prng(1);
	std::normal_distribution<double> normal;
	auto randn = std::bind(normal, prng);

	double mu    = 5.0;
	double sigma = 3.0;
	std::cout << "mu = " << mu << ", sigma = " << sigma << std::endl;

	Function f;
	f.add_variable(&mu, 1);
	f.add_variable_with_change<GreaterThanZero>(&sigma, 1, 1);

	for (int i = 0; i < 10000; ++i) {
		double sample = sigma*randn() + mu;
		f.add_term(std::make_shared<IntervalTerm<NegLogLikelihood, 1, 1>>(sample), &mu, &sigma);
	}

	mu    = 0.0;
	sigma = 1.0;
	Solver solver;
	SolverResults results;
	solver.solve_lbfgs(f, &results);
	std::cout << "Estimated:" << std::endl;
	std::cout << "mu = " << mu << ", sigma = " << sigma << std::endl << std::endl;

	// Remove the constraint on sigma.
	f.add_variable(&sigma, 1);

	Interval<double> start_mu(4.0, 6.0);
	Interval<double> start_sigma(1.0, 10.0);
	IntervalVector start_box;
	start_box.push_back(start_mu);
	start_box.push_back(start_sigma);
	solver.maximum_iterations = 5000;
	solver.solve_global(f, start_box, &results);
	std::cout << results << std::endl;

	std::cout << "Global optimization:" << std::endl;
	std::cout << "mu = " << mu << ", sigma = " << sigma << std::endl << std::endl;

	return 0;
}

int main()
{
	try {
		return main_function();
	}
	catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
		return 1;
	}
}