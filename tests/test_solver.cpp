
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include <spii/solver.h>

struct Banana
{
	template<typename R>
	R operator()(const R* const x)
	{
		R d0 =  x[1] - x[0]*x[0];
		R d1 =  1 - x[0];
		return 100 * d0*d0 + d1*d1;
	}
};

TEST(Solver, banana)
{
	Function f;
	double x[3] = {-1.2, 1.0};
	f.add_variable(x, 2);
	f.add_term(new AutoDiffTerm<Banana, 2>(new Banana()), x);

	Solver solver;
	solver.maximum_iterations = 50;
	SolverResults results;
	solver.Solve(f, &results);

	EXPECT_LT( std::abs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::abs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::abs(f.evaluate()), 1e-9);
}

struct NegLogLikelihood
{
	double sample;
	NegLogLikelihood(double sample)
	{
		this->sample = sample;
	}

	template<typename R>
	R operator()(const R* const mu, const R* const logsigma)
	{
		R diff = (*mu - sample) / exp(*logsigma);
		return 0.5 * diff*diff + *logsigma;
	}
};

TEST(Solver, gaussian)
{
	std::mt19937 prng(0);
	std::normal_distribution<double> normal;
	std::variate_generator<std::tr1::mt19937, std::tr1::normal_distribution<double> > randn(prng,normal);

	double mu    = 5.0;
	double sigma = 3.0;
	double logsigma = std::log(sigma);
	Function f;
	f.add_variable(&mu, 1);
	f.add_variable(&logsigma, 1);
	for (int i = 0; i < 1000; ++i) {
		NegLogLikelihood* llh = new NegLogLikelihood(sigma*randn() + mu);
		f.add_term(new AutoDiffTerm<NegLogLikelihood, 1, 1>(llh), &mu, &logsigma);
	}
	mu       = 0.0;
	logsigma = 0.0;

	Solver solver;
	solver.maximum_iterations = 50;
	SolverResults results;
	solver.Solve(f, &results);

	sigma = std::exp(logsigma);
	std::cout << "mu = " << mu << ", sigma = " << sigma << std::endl;
	
	// Expect the parameters to be roughly correct.
	EXPECT_LT( std::abs(mu - 5.0) / 5.0, 0.05);
	EXPECT_LT( std::abs(sigma - 3.0) / 3.0, 0.05);
}
