// Petter Strandmark 2012.

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include <spii/spii.h>
#include <spii/solver.h>

namespace spii {

// Holds a point in the Nelder-Mead simplex.
// Equipped with a comparison operator for sorting.
struct SimplexPoint
{
	Eigen::VectorXd x;
	double value;

	bool operator<(const SimplexPoint& rhs) const
	{
		return this->value < rhs.value;
	}
};

// If required for debugging.
std::ostream& operator<<(std::ostream& out, const SimplexPoint& point)
{
	out << point.x.transpose() << " : " << point.value;
	return out;
}

}  // namespace spii

namespace std
{
	template<>
	void swap<spii::SimplexPoint>(spii::SimplexPoint& lhs, spii::SimplexPoint& rhs)
	{
		lhs.x.swap(rhs.x);
		swap(lhs.value, rhs.value);
	}
}

namespace spii {

void Solver::solve_nelder_mead(const Function& function,
                               SolverResults* results) const
{
	double global_start_time = wall_time();

	// Dimension of problem.
	size_t n = function.get_number_of_scalars();

	// The Nelder-Mead simplex.
	std::vector<SimplexPoint> simplex(n + 1);

	// Copy the user state to the current point.
	Eigen::VectorXd x;
	function.copy_user_to_global(&x);

	// TODO: Find a better initialization.
	for (size_t i = 0; i < n + 1; ++i) {
		simplex[i].x = x;
		if (i < n) {
			if (std::abs(x[i]) < 0.025) {
				simplex[i].x[i] += 0.05;
			}
			else {
				simplex[i].x[i] += 0.05 * x[i];
			}
		}

		simplex[i].value = function.evaluate(simplex[i].x);
	}

	std::sort(simplex.begin(), simplex.end());

	SimplexPoint mean_point;
	SimplexPoint reflection_point;
	SimplexPoint expansion_point;
	mean_point.x.resize(n);
	reflection_point.x.resize(n);
	expansion_point.x.resize(n);

	double fmin  = std::numeric_limits<double>::quiet_NaN();
	double fmax  = std::numeric_limits<double>::quiet_NaN();
	double fval  = std::numeric_limits<double>::quiet_NaN();
	double fprev = std::numeric_limits<double>::quiet_NaN();
	double area0 = std::numeric_limits<double>::quiet_NaN();

	Eigen::MatrixXd area_mat(n, n);

	//
	// START MAIN ITERATION
	//
	results->startup_time   = wall_time() - global_start_time;
	results->exit_condition = SolverResults::ERROR;
	int iter = 0;
	while (true) {

		//
		// In each iteration, the worst point in the simplex
		// is replaced with a new one.
		//
		double start_time = wall_time();

		mean_point.x.setZero();
		fval = 0;
		// Compute the mean of the best n points.
		for (size_t i = 0; i < n; ++i) {
			mean_point.x += simplex[i].x;
			fval         += simplex[i].value;
		}
		fval         /= double(n);
		mean_point.x /= double(n);
		fmin = simplex[0].value;
		fmax = simplex[n].value;

		// Compute the area of the simplex.
		for (size_t i = 0; i < n; ++i) {
			area_mat.col(i) = simplex[i].x - simplex[n].x;
		}
		double area = std::abs(area_mat.determinant());
		if (iter == 0) {
			area0 = area;
		}

		const char* iteration_type = "n/a";

		// Compute the reflexion point and evaluate it.
		reflection_point.x = 2.0 * mean_point.x - simplex[n].x;
		reflection_point.value = function.evaluate(reflection_point.x);

		if (simplex[0].value <= reflection_point.value &&
			reflection_point.value < simplex[n - 1].value) {
			// Reflected point is neither better nor worst in the
			// new simplex.
			std::swap(reflection_point, simplex[n]);
			iteration_type = "Reflect 1";
		}
		else if (reflection_point.value < simplex[0].value) {
			// Reflected point is better than the current best; try
			// to go farther along this direction.

			// Compute expansion point.
			expansion_point.x = 3.0 * mean_point.x - 2.0 * simplex[n].x;
			expansion_point.value = function.evaluate(expansion_point.x);

			if (expansion_point.value < reflection_point.value) {
				std::swap(expansion_point, simplex[n]);
				iteration_type = "Expansion";
			}
			else {
				std::swap(reflection_point, simplex[n]);
				iteration_type = "Reflect 2";
			}
		}
		else {
			// Reflected point is still worse than x[n]; contract.
			bool success = false;

			if (simplex[n - 1].value <= reflection_point.value &&
			    reflection_point.value < simplex[n].value) {
				// Try to perform "outside" contraction.
				expansion_point.x = 1.5 * mean_point.x - 0.5 * simplex[n].x;
				expansion_point.value = function.evaluate(expansion_point.x);

				if (expansion_point.value <= reflection_point.value) {
					std::swap(expansion_point, simplex[n]);
					success = true;
					iteration_type = "Outside contraction";
				}
			}
			else {
				// Try to perform "inside" contraction.
				expansion_point.x = 0.5 * mean_point.x + 0.5 * simplex[n].x;
				expansion_point.value = function.evaluate(expansion_point.x);

				if (expansion_point.value < simplex[n].value) {
					std::swap(expansion_point, simplex[n]);
					success = true;
					iteration_type = "Inside contraction";
				}
			}

			if (! success) {
				// Neither outside nor inside contraction was acceptable;
				// shrink the simplex toward the best point.
				for (size_t i = 1; i < n + 1; ++i) {
					simplex[i].x = 0.5 * (simplex[0].x + simplex[i].x);
					simplex[i].value = function.evaluate(simplex[i].x);
					iteration_type = "Shrink";
				}
			}
		}

		std::sort(simplex.begin(), simplex.end());

		results->function_evaluation_time += wall_time() - start_time;

		//
		// Test stopping criteriea
		//
		start_time = wall_time();
		if (area / area0 < this->gradient_tolerance) {
			results->exit_condition = SolverResults::GRADIENT_TOLERANCE;
			break;
		}
		if (iter >= this->maximum_iterations) {
			results->exit_condition = SolverResults::NO_CONVERGENCE;
			break;
		}
		results->stopping_criteria_time += wall_time() - start_time;

		//
		// Log the results of this iteration.
		//
		start_time = wall_time();

		int log_interval = 1;
		if (iter > 30) {
			log_interval = 10;
		}
		if (iter > 200) {
			log_interval = 100;
		}
		if (iter > 2000) {
			log_interval = 1000;
		}
		if (this->log_function && iter % log_interval == 0) {
			char str[1024];
				if (iter == 0) {
					this->log_function("Itr     min(f)     avg(f)     max(f)    area     type");
				}
				std::sprintf(str, "%4d %+.3e %+.3e %+.3e %.3e %s",
					iter, fmin, fval, fmax, area, iteration_type);
			this->log_function(str);
		}
		results->log_time += wall_time() - start_time;

		fprev = fmax;
		iter++;
	}

	// Return the best point as solution.
	function.copy_global_to_user(simplex[0].x);
	results->total_time = wall_time() - global_start_time;

	if (this->log_function) {
		char str[1024];
		std::sprintf(str, " end %+.3e", fval);
		this->log_function(str);
	}
}

}  // namespace spii
