
#include <sstream>
#include <stdexcept>

#ifdef USE_OPENMP
	#include <omp.h>
#endif

#include <spii/spii.h>

namespace spii
{

double wall_time()
{
	#ifdef USE_OPENMP
		return omp_get_wtime();
	#else
		return 0.0;
	#endif
}

void check(bool expr, const char* message)
{
	if (!expr) {
		throw std::invalid_argument(message);
	}
}

void assertion_failed(const char* expr, const char* file, int line)
{
	std::stringstream sout;
	sout << "Assertion failed: " << expr << " in " << file << ":" << line << ".";
	throw std::runtime_error(sout.str().c_str());
}

}
