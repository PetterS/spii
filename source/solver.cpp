
#include <stdexcept>

#include <spii/solver.h>

namespace spii {

SolverResults::SolverResults()
{
	this->exit_condition = NA;

	this->startup_time              = 0;
	this->function_evaluation_time  = 0;
	this->stopping_criteria_time    = 0;
	this->matrix_factorization_time = 0;
	this->lbfgs_update_time         = 0;
	this->linear_solver_time        = 0;
	this->backtracking_time         = 0;
	this->log_time                  = 0;
	this->total_time                = 0;
}

std::ostream& operator<<(std::ostream& out, const SolverResults& results)
{
	out << "----------------------------------------------\n";
	out << "Exit condition            : ";
	#define EXIT_ENUM_IF(val) if (results.exit_condition == SolverResults::val) out << #val << '\n';
	EXIT_ENUM_IF(GRADIENT_TOLERANCE);
	EXIT_ENUM_IF(FUNCTION_TOLERANCE);
	EXIT_ENUM_IF(ARGUMENT_TOLERANCE);
	EXIT_ENUM_IF(NO_CONVERGENCE);
	EXIT_ENUM_IF(FUNCTION_NAN);
	EXIT_ENUM_IF(FUNCTION_INFINITY);
	EXIT_ENUM_IF(ERROR);
	EXIT_ENUM_IF(NA);
	out << "----------------------------------------------\n";
	out << "Startup time              : " << results.startup_time << '\n';
	out << "Function evaluation time  : " << results.function_evaluation_time << '\n';
	out << "Stopping criteria time    : " << results.stopping_criteria_time << '\n';
	out << "Matrix factorization time : " << results.matrix_factorization_time << '\n';
	out << "L-BFGS update time        : " << results.lbfgs_update_time << '\n';
	out << "Linear solver time        : " << results.linear_solver_time << '\n';
	out << "Backtracking time         : " << results.backtracking_time << '\n';
	out << "Log time                  : " << results.log_time << '\n';
	out << "Total time (without log)  : " << results.total_time - results.log_time << '\n';
	out << "----------------------------------------------\n";
	return out;
}

Solver::Solver()
{
	this->sparsity_mode = AUTO;
	this->log_function = cerr_log_function;
	this->maximum_iterations = 100;
	this->gradient_tolerance = 1e-12;
	this->function_improvement_tolerance = 1e-12;
	this->argument_improvement_tolerance = 1e-12;
	this->area_tolerance = 1e-12;

	this->lbfgs_history_size = 10;
	this->lbfgs_restart_tolerance = 1e-6;

	#ifdef _MSC_VER
		_set_output_format(_TWO_DIGIT_EXPONENT);
	#endif
}

void Solver::solve(const Function& function,
                   Method method,
                   SolverResults* results) const
{
	if (method == NEWTON) {
		solve_newton(function, results);
	}
	else if (method == LBFGS) {
		solve_lbfgs(function, results);
	}
	else if (method == NELDER_MEAD) {
		solve_nelder_mead(function, results);
	}
	else if (method == PATTERN_SEARCH) {
		solve_pattern_search(function, results);
	}
	else {
		throw std::runtime_error("Solver::solve: unknown method.");
	}
}

}  // namespace spii

