
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

void assertion_failed(const char* expr, const char* file_cstr, int line)
{
	using namespace std;

	// Extract the file name only.
	string file(file_cstr);
	auto pos = file.find_last_of("/\\");
	if (pos == string::npos) {
		pos = 0;
	}
	file = file.substr(pos + 1);  // Returns empty string if pos + 1 == length.

	stringstream sout;
	sout << "Assertion failed: " << expr << " in " << file << ":" << line << ".";
	throw runtime_error(sout.str().c_str());
}

}
