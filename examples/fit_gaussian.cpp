// Petter Strandmark 2012--2013.

#include <functional>
#include <iostream>
#include <random>

#include <spii/auto_diff_term.h>
#include <spii/interval_term.h>
#include <spii/solver.h>
#include <spii/transformations.h>

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
	using namespace std;

	mt19937 prng(1u);
	normal_distribution<double> normal;
	auto randn = bind(normal, prng);

	double mu    = 5.0;
	double sigma = 3.0;
	cout << "mu = " << mu << ", sigma = " << sigma << endl;

	Function f;
	f.add_variable(&mu, 1);
	f.add_variable_with_change<GreaterThanZero>(&sigma, 1, 1);

	for (int i = 0; i < 10000; ++i) {
		double sample = sigma*randn() + mu;
		f.add_term(std::make_shared<IntervalTerm<NegLogLikelihood, 1, 1>>(sample), &mu, &sigma);
	}

	mu    = 0.0;
	sigma = 1.0;
	LBFGSSolver solver;

	solver.callback_function = [&](const CallbackInformation& information) -> bool
	{
		f.copy_global_to_user(*information.x);
		cerr << "  -- mu = " << mu << ", sigma = " << sigma << endl;
		return true;
	};

	SolverResults results;
	solver.solve(f, &results);
	cout << "Estimated:" << endl;
	cout << "f = " << f.evaluate() << " mu = " << mu << ", sigma = " << sigma << endl << endl;

	cout << "Perform global optimization? (y/n):";
	char answer = 'n';
	cin >> answer;
	if (cin && tolower(answer) == 'y') {
		sigma = 3.0;
		f.set_constant(&sigma, true);

		GlobalSolver global_solver;
		std::vector<Interval<double>> mu_interval;
		mu_interval.push_back(Interval<double>(-10.0, 10.0));
		global_solver.maximum_iterations = 1000;
		auto interval = global_solver.solve_global(f, mu_interval, &results);
		cout << "Optimal parameter interval (sigma is kept at " << sigma << "):" << endl;
		cout << "   mu    = " << interval.at(0) << endl;
	}
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
