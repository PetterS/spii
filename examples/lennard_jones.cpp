// Petter Strandmark 2012.
//
// See http://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html
// for best known minima for N <= 150.
//

#include <functional>
#include <iomanip>
#include <iostream>
#include <random>

#include <spii/auto_diff_term.h>
#include <spii/solver.h>

using namespace spii;

struct LennardJonesTerm
{
	template<typename R>
	R operator()(const R* const p1, const R* const p2) const
	{
		R dx = p1[0] - p2[0];
		R dy = p1[1] - p2[1];
		R dz = p1[2] - p2[2];
		R r2 = dx*dx + dy*dy + dz*dz;
		R r6  = r2*r2*r2;
		R r12 = r6*r6;
		return 1.0 / r12 - 2.0 / r6;
	}
};

int main()
{
	std::mt19937 prng(0);
	std::normal_distribution<double> normal;
	auto randn = std::bind(normal, prng);

	size_t N = -1;
	std::cout << "Enter N = ";
	std::cin >> N;

	Function potential;
	std::vector<Eigen::Vector3d> points(N);

	int n = std::ceil(std::pow(double(N), 1.0/3.0));

	// Initial position is a cubic grid with random pertubations.
	for (size_t i = 0; i < N; ++i) {
		int x =  i % n;
		int y = (i / n) % n;
		int z = (i / n) / n;

		potential.add_variable(&points[i][0], 3);
		points[i][0] = x + 0.05 * randn();
		points[i][1] = y + 0.05 * randn();
		points[i][2] = z + 0.05 * randn();
	}

	for (size_t i = 0; i < N; ++i) {
		for (int j = i + 1; j < N; ++j) {
			potential.add_term(
				new AutoDiffTerm<LennardJonesTerm, 3, 3>(
					new LennardJonesTerm),
					&points[i][0],
					&points[j][0]);
		}
	}

	Solver solver;
	solver.sparsity_mode = Solver::DENSE;
	solver.maximum_iterations = 100;
	SolverResults results;
	solver.solve_newton(potential, &results);

	std::cerr << results;
	potential.print_timing_information(std::cout);

	std::cout << "Energy = " << std::setprecision(10) << potential.evaluate() << std::endl;
}