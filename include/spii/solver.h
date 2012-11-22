#ifndef SPII_SOLVER_H
#define SPII_SOLVER_H

#include <string>

#include <spii/function.h>

struct SolverResults
{
};

void cerr_log_function(const std::string& log_message);

class Solver
{
public:
	Solver();
	void Solve(const Function& function,
	           SolverResults* results) const;

	// Function called each iteration with a log message.
	void (*log_function)(const std::string& log_message);
	// Maximum number of iterations. Default: 100.
	int maximum_iterations;
};

#endif
