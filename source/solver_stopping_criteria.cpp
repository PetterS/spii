

#include <spii/solver.h>

namespace spii {

bool Solver::check_exit_conditions(const double fval,
                                   const double fprev,
                                   const double normg,
                                   const double normg0,
                                   const double normx,
                                   const double normdx,
                                   SolverResults* results) const
{
	if (fval != fval) {
		// NaN encountered.
		if (this->log_function) {
			this->log_function("f(x) is NaN.");
		}
		results->exit_condition = SolverResults::FUNCTION_NAN;
		return true;
	}

	if (normg / normg0 < this->gradient_tolerance) {
	    results->exit_condition = SolverResults::GRADIENT_TOLERANCE;
		return true;
	}

	if (abs(fval - fprev) / (abs(fval) + this->function_improvement_tolerance) <
	                                     this->function_improvement_tolerance) {
		results->exit_condition = SolverResults::FUNCTION_TOLERANCE;
		return true;
	}

	if (normdx / (normx + this->argument_improvement_tolerance) <
	                      this->argument_improvement_tolerance) {
		results->exit_condition = SolverResults::ARGUMENT_TOLERANCE;
		return true;
	}

	if (fval ==  std::numeric_limits<double>::infinity() ||
		fval == -std::numeric_limits<double>::infinity()) {
		// Infinity encountered.
		if (this->log_function) {
			this->log_function("f(x) is infinity.");
		}
		results->exit_condition = SolverResults::FUNCTION_INFINITY;
		return true;
	}
	return false;
}

}  // namespace spii
