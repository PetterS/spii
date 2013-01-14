// Petter Strandmark 2012.

#include <functional>
#include <iostream>
#include <random>

#include <spii/auto_diff_term.h>
#include <spii/constraints.h>
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
	std::mt19937 prng(0);
	std::normal_distribution<double> normal;
	auto randn = std::bind(normal, prng);

	double mu    = 5.0;
	double sigma = 3.0;
	std::cout << "mu = " << mu << ", sigma = " << sigma << std::endl;

	Function f;
	f.add_variable(&mu, 1);
	f.add_variable(&sigma, 1, new GreaterThanZero(1));

	for (int i = 0; i < 10000; ++i) {
		double sample = sigma*randn() + mu;
		auto* llh = new NegLogLikelihood(sample);
		f.add_term(new AutoDiffTerm<NegLogLikelihood, 1, 1>(llh), &mu, &sigma);
	}

	mu    = 0.0;
	sigma = 1.0;
	Solver solver;
	SolverResults results;
	solver.solve_lbfgs(f, &results);
	std::cout << "Estimated:" << std::endl;
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