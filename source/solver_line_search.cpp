// Petter Strandmark 2012.

#include <Eigen/Dense>

#include <spii/solver.h>

namespace spii {

double Solver::perform_linesearch(const Function& function,
                                  const Eigen::VectorXd& x,
                                  const double fval,
                                  const Eigen::VectorXd& g,
                                  const Eigen::VectorXd& p,
                                  Eigen::VectorXd* scratch,
                                  const double start_alpha) const
{
	//
	// Perform back-tracking line search.
	//

	// Starting value for alpha during line search. Newton and
	// quasi-Newton methods should choose 1.0.
	double alpha = start_alpha;
	double rho = this->line_search_rho;
	double c = this->line_search_c;
	double gTp = g.dot(p);
	if (gTp != gTp) {
		if (this->log_function) {
			this->log_function("Backtracking encountered NaN, returning zero step.");
		}
		return 0.0;
	}

	int backtracking_attempts = 0;
	while (true) {
		*scratch = x + alpha * p;
		double lhs = function.evaluate(*scratch);
		double rhs = fval + c * alpha * gTp;
		if (lhs <= rhs) {
			break;
		}
		alpha *= rho;

		backtracking_attempts++;
		if (backtracking_attempts > 1000) {
			if (this->log_function) {
				this->log_function("Backtracking failed, returning zero step.");
			}
			return 0.0;
		}
	}

	return alpha;
}

}  // namespace spii
