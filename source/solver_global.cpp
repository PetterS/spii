// Petter Strandmark 2013.

#include <queue>
#include <set>
#include <sstream>

#include <spii/solver.h>
#include <spii/spii.h>

namespace spii {

struct GlobalQueueEntry
{
	double volume;
	IntervalVector box;
	double best_known_lower_bound;

	bool operator < (const GlobalQueueEntry& rhs) const
	{
		return this->volume < rhs.volume;
	}
};

typedef std::priority_queue<GlobalQueueEntry> IntervalQueue;

void midpoint(const IntervalVector& x, Eigen::VectorXd* x_mid)
{
	x_mid->resize(x.size());
	for (int i = 0; i < x.size(); ++i) {
		(*x_mid)[i] = (x[i].get_upper() + x[i].get_lower()) / 2.0;
	}
}

double volume(const IntervalVector& x)
{
	double vol = 1.0;
	for (auto itr = x.begin(); itr != x.end(); ++itr) {
		const auto& interval = *itr;
		vol *= interval.get_upper() - interval.get_lower();
	}
	return vol;
}

IntervalVector get_bounding_box(const IntervalQueue& queue_in,
                                double* function_lower_bound,
                                double* volume)
{
	if (queue_in.empty()) {
		return IntervalVector();
	}
	// Make copy.
	auto queue = queue_in;
	auto n = queue_in.top().box.size();
	std::vector<double> upper_bound(n, -1e100);
	std::vector<double> lower_bound(n, 1e100);

	*function_lower_bound = std::numeric_limits<double>::infinity();
	*volume = 0;
	while (!queue.empty()) {
		const auto& box = queue.top().box;
		for (int i = 0; i < n; ++i) {
			lower_bound[i] = std::min(lower_bound[i], box[i].get_lower());
			upper_bound[i] = std::max(upper_bound[i], box[i].get_upper());
		}
		*volume += spii::volume(box);
		*function_lower_bound = std::min(*function_lower_bound, queue.top().best_known_lower_bound);
		queue.pop();
	}

	IntervalVector out(n);
	for (int i = 0; i < n; ++i) {
		out[i] = Interval<double>(lower_bound[i], upper_bound[i]);
	}

	return out;
}

void split_interval(const IntervalVector& x,
                    double lower_bound,
                    IntervalQueue* queue)
{
	auto n = x.size();
	std::vector<int> split(n, 0);

	Eigen::VectorXd mid;
	midpoint(x, &mid);

	while (true) {

		IntervalVector x_split(n);
		double volume = 1.0;
		for (int i = 0; i < n; ++i) {
			double a, b;
			if (split[i] == 0) {
				a = x[i].get_lower();
				b = mid[i];
			}
			else {
				a = mid[i];
				b = x[i].get_upper();
			}
			x_split[i] = Interval<double>(a, b);
			volume *= b - a;
		}

		GlobalQueueEntry entry;
		entry.volume = volume;
		entry.box = x_split;
		entry.best_known_lower_bound = lower_bound;
		queue->push(entry);

		// Move to the next binary vector
		//  000001
		//  000010
		//  000011
		//   ...
		//  111111
		int i = 0;
		split[i]++;
		while (split[i] > 1) {
			split[i] = 0;
			i++;
			if (i < n) {
				split[i]++;
			}
			else {
				break;
			}
		}
		if (i == n) {
			break;
		}
	}
}

void GlobalSolver::solve(const Function& function,
                         SolverResults* results) const
{
	throw std::runtime_error("GlobalSolver::solve_global should be called.");
}

void GlobalSolver::solve_global(const Function& function,
                                const IntervalVector& x_interval,
                                SolverResults* results) const
{
	using namespace std;
	double global_start_time = wall_time();

	/*
	for (auto itr = function.variables.begin(); itr != function.variables.end(); ++itr) {
		if (itr->second.change_of_variables != 0) {
			throw std::runtime_error("Solver::solve_global: "
				"function can not have change of variables.");
		}
	}*/

	IntervalQueue queue;
	GlobalQueueEntry entry;
	entry.volume = 1e100;
	entry.box = x_interval;
	queue.push(entry);

	auto bounds = function.evaluate(x_interval);
	double upper_bound = bounds.get_upper();
	double lower_bound = - std::numeric_limits<double>::infinity();

	Eigen::VectorXd best_x(x_interval.size());

	int iterations = 0;
	results->exit_condition = SolverResults::INTERNAL_ERROR;
	while (!queue.empty()) {
		double start_time = wall_time();

		if (iterations >= this->maximum_iterations) {
			results->exit_condition = SolverResults::NO_CONVERGENCE;
			break;
		}

		const auto box = queue.top().box;
		// Remove current element from queue.
		queue.pop();

		auto bounds = function.evaluate(box);

		if (bounds.get_lower() < upper_bound) {
			// Evaluate middle point.
			Eigen::VectorXd x(box.size());
			midpoint(box, &x);
			double value = function.evaluate(x);

			if (value < upper_bound) {
				upper_bound = value;
				best_x = x;
			}

			// Add new elements to queue.
			split_interval(box, bounds.get_lower(), &queue);
		}
		results->function_evaluation_time += wall_time() - start_time;

		iterations++;

		start_time = wall_time();
		int log_interval = 1;
		if (iterations > 20) {
			log_interval = 10;
		}
		if (iterations > 200) {
			log_interval = 100;
		}
		if (iterations > 2000) {
			log_interval = 1000;
		}
		if (iterations > 20000) {
			log_interval = 10000;
		}
		if (iterations > 200000) {
			log_interval = 100000;
		}
		if (iterations >= this->maximum_iterations - 2) {
			log_interval = 1;
		}

		if (iterations % log_interval == 0) {
			double vol_sum;
			auto bounding_box = get_bounding_box(queue, &lower_bound, &vol_sum);
			double vol_bounding = volume(bounding_box);

			double avg_magnitude = (std::abs(lower_bound) + std::abs(upper_bound)) / 2.0;
			double relative_gap = (upper_bound - lower_bound) / avg_magnitude;

			results->stopping_criteria_time += wall_time() - start_time;
			start_time = wall_time();

			if (this->log_function) {
				if (iterations == 1) {
					this->log_function("Iteration Q-size   l.bound   u.bound     rel.gap    bounding   volume");
					this->log_function("----------------------------------------------------------------------");
				}

				char tmp[1024];
				sprintf(tmp, "%9d %6d %+10.3e %+10.3e %10.2e %10.3e %10.3e",
					iterations,
					int(queue.size()),
					lower_bound,
					upper_bound,
					relative_gap,
					vol_bounding,
					vol_sum);
				this->log_function(tmp);
			}

			results->log_time += wall_time() - start_time;

			if (relative_gap <= this->function_improvement_tolerance) {
				results->exit_condition = SolverResults::FUNCTION_TOLERANCE;
				break;
			}

			if (vol_bounding <= this->argument_improvement_tolerance) {
				results->exit_condition = SolverResults::ARGUMENT_TOLERANCE;
				break;
			}
		}
	}

	function.copy_global_to_user(best_x);

	results->optimum_lower = lower_bound;
	results->optimum_upper = upper_bound;
	results->total_time = wall_time() - global_start_time;
}

}  // namespace spii

