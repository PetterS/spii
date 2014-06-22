// Petter Strandmark 2013.

#include <algorithm>
#include <cstdio>
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

typedef std::vector<GlobalQueueEntry> IntervalQueue;

std::ostream& operator << (std::ostream& out, IntervalQueue queue)
{
	while (!queue.empty()) {
		const auto box = queue.front().box;
		std::pop_heap(begin(queue), end(queue)); queue.pop_back();
		out << box[0] << "; ";
	}
	out << std::endl;
	return out;
}

void midpoint(const IntervalVector& x, Eigen::VectorXd* x_mid)
{
	x_mid->resize(x.size());
	for (int i = 0; i < x.size(); ++i) {
		(*x_mid)[i] = (x[i].get_upper() + x[i].get_lower()) / 2.0;
	}
}

double volume(const IntervalVector& x)
{
	if (x.empty()) {
		return 0.0;
	}

	double vol = 1.0;
	for (auto itr = x.begin(); itr != x.end(); ++itr) {
		const auto& interval = *itr;
		vol *= interval.get_upper() - interval.get_lower();
	}
	return vol;
}

IntervalVector get_bounding_box(const IntervalQueue& queue_in,
                                double* function_lower_bound,
                                double* sum_of_volumes)
{
	*sum_of_volumes = 0;

	if (queue_in.empty()) {
		return IntervalVector();
	}

	auto n = queue_in.front().box.size();
	std::vector<double> upper_bound(n, -1e100);
	std::vector<double> lower_bound(n, 1e100);

	*function_lower_bound = std::numeric_limits<double>::infinity();

	for (const auto& elem: queue_in) {
		const auto& box = elem.box;
		for (int i = 0; i < n; ++i) {
			lower_bound[i] = std::min(lower_bound[i], box[i].get_lower());
			upper_bound[i] = std::max(upper_bound[i], box[i].get_upper());
		}
		*sum_of_volumes += spii::volume(box);
		*function_lower_bound = std::min(*function_lower_bound, elem.best_known_lower_bound);
	}

	IntervalVector out(n);
	for (int i = 0; i < n; ++i) {
		out[i] = Interval<double>(lower_bound[i], upper_bound[i]);
	}

	return out;
}

//
// Splits an interval into 2^n subintervals and adds them all to the
// queue.
//
void split_interval(const IntervalVector& x,
                    double lower_bound,
                    IntervalQueue* queue)
{
	auto n = x.size();
	std::vector<int> split(n, 0);

	Eigen::VectorXd mid;
	midpoint(x, &mid);

	while (true) {

		queue->emplace_back();
		GlobalQueueEntry& entry = queue->back();
		IntervalVector& x_split = entry.box;
		x_split.resize(n);

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

		entry.volume = volume;
		entry.best_known_lower_bound = lower_bound;
		std::push_heap(begin(*queue), end(*queue));

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

//
// Splits an interval into two along its largest dimension.
//
void split_interval_single(const IntervalVector& x,
                           double lower_bound,
                           IntervalQueue* queue)
{
	auto n = x.size();
	size_t max_index = 0;
	double max_length = -1;
	for (size_t i = 0; i < x.size(); ++i) {
		if (x[i].length() > max_length) {
			max_index = i;
			max_length = x[i].length();
		}
	}

	for (int i = 0; i <= 1; ++i) {
		queue->emplace_back();
		GlobalQueueEntry& entry = queue->back();
		entry.box = x;
		if (i == 0) {
			entry.box[max_index] = Interval<double>(x[max_index].get_lower(), x[max_index].get_lower() + max_length / 2.0);
		}
		else {
			entry.box[max_index] = Interval<double>(x[max_index].get_lower() + max_length / 2.0, x[max_index].get_upper());
		}

		entry.volume = volume(entry.box);
		entry.best_known_lower_bound = lower_bound;

		std::push_heap(begin(*queue), end(*queue));
	}
}

void GlobalSolver::solve(const Function& function,
                         SolverResults* results) const
{
	spii_assert(false, "GlobalSolver::solve_global should be called.");
}

IntervalVector GlobalSolver::solve_global(const Function& function,
                                          const IntervalVector& x_interval,
                                          SolverResults* results) const
{
	using namespace std;
	double global_start_time = wall_time();

	check(x_interval.size() == function.get_number_of_scalars(),
		"solve_global: input vector does not match the function's number of scalars");
	auto n = x_interval.size();

	IntervalQueue queue;
	queue.reserve(2 * this->maximum_iterations);

	GlobalQueueEntry entry;
	entry.volume = 1e100;
	entry.box = x_interval;
	queue.push_back(entry);

	auto bounds = function.evaluate(x_interval);
	double upper_bound = bounds.get_upper();
	double lower_bound = - std::numeric_limits<double>::infinity();

	Eigen::VectorXd best_x(n);
	IntervalVector best_interval;

	int iterations = 0;
	results->exit_condition = SolverResults::INTERNAL_ERROR;

	while (!queue.empty()) {
		double start_time = wall_time();

		if (iterations >= this->maximum_iterations) {
			results->exit_condition = SolverResults::NO_CONVERGENCE;
			break;
		}

		const auto box = queue.front().box;
		best_interval = box;
		// Remove current element from queue.
		pop_heap(begin(queue), end(queue)); queue.pop_back();

		auto bounds = function.evaluate(box);

		//cerr << "-- Processing " << box << " resulting in " << bounds << ". Upper bound is " << upper_bound << endl;

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
			//split_interval(box, bounds.get_lower(), &queue);
			split_interval_single(box, bounds.get_lower(), &queue);
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
			double volumes_sum;
			lower_bound = upper_bound;  // Default lower bound if queue is empty (problem is solved).
			auto bounding_box = get_bounding_box(queue, &lower_bound, &volumes_sum);
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
					volumes_sum);
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

	double tmp1, tmp2;
	auto bounding_box = get_bounding_box(queue, &tmp1, &tmp2);
	if (bounding_box.empty()) {
		// Problem was solved exactly (queue empty)
		spii_assert(queue.empty());
		results->exit_condition = SolverResults::FUNCTION_TOLERANCE;
		bounding_box = best_interval;
	}

	function.copy_global_to_user(best_x);

	results->optimum_lower = lower_bound;
	results->optimum_upper = upper_bound;
	results->total_time = wall_time() - global_start_time;

	return bounding_box;
}

}  // namespace spii

