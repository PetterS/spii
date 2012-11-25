
#include <functional>
#include <iostream>
#include <random>

#include <spii/auto_diff_term.h>
#include <spii/solver.h>

struct NegLogLikelihood
{
	double sample;
	NegLogLikelihood(double sample)
	{
		this->sample = sample;
	}

	template<typename R>
	R operator()(const R* const mu, const R* const logsigma) const
	{
		R diff = (*mu - sample) / exp(*logsigma);
		return 0.5 * diff*diff + *logsigma;
	}
};

int main()
{
	std::mt19937 prng(0);
	std::normal_distribution<double> normal;
	auto randn = std::bind(normal, prng);

	double mu    = 5.0;
	double sigma = 3.0;
	std::cout << "mu = " << mu << ", sigma = " << sigma << std::endl;
	double logsigma = std::log(sigma);
	Function f;
	f.add_variable(&mu, 1);
	f.add_variable(&logsigma, 1);
	for (int i = 0; i < 10000; ++i) {
		NegLogLikelihood* llh = new NegLogLikelihood(sigma*randn() + mu);
		f.add_term(new AutoDiffTerm<NegLogLikelihood, 1, 1>(llh), &mu, &logsigma);
	}
	mu       = 0.0;
	logsigma = 0.0;

	Solver solver;
	solver.maximum_iterations = 50;
	SolverResults results;
	solver.Solve(f, &results);

	std::cerr << results;
	f.print_timing_information(std::cerr);

	sigma = std::exp(logsigma);
	std::cout << "Estimated:" << std::endl;
	std::cout << "mu = " << mu << ", sigma = " << sigma << std::endl;
}
