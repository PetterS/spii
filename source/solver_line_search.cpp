
#include <Eigen/Dense>

#include <spii/solver.h>

double Solver::perform_linesearch(const Function& function,
                                  const Eigen::VectorXd& x,
                                  const double fval,
								  const Eigen::VectorXd& g,
                                  const Eigen::VectorXd& p,
                                  Eigen::VectorXd* scratch) const
{
	//
	// Perform back-tracking line search.
	//

	// Starting value for alpha during line search. Newton and
	// quasi-Newton methods should choose 1.0.
	double alpha = 1.0;
	double rho = 0.5;
	double c = 0.5;
	double gTp = g.dot(p);
	int backtracking_attempts = 0;
	while (true) {
		*scratch = x + alpha * p;
		double lhs = function.evaluate(*scratch);
		double rhs = fval + c * alpha * g.dot(p);
		if (lhs <= rhs) {
			break;
		}
		alpha *= rho;

		backtracking_attempts++;
		if (backtracking_attempts > 100) {
			throw std::runtime_error("Solver::solve: Backtracking failed.");
		}
	}

	return alpha;
}
