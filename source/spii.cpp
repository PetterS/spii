
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


// Removes the path from __FILE__ constants and keeps the name only.
std::string extract_file_name(const char* full_file_cstr)
{
	using namespace std;

	// Extract the file name only.
	string file(full_file_cstr);
	auto pos = file.find_last_of("/\\");
	if (pos == string::npos) {
		pos = 0;
	}
	file = file.substr(pos + 1);  // Returns empty string if pos + 1 == length.

	return file;
}

void verbose_error_internal(const char* expr, const char* full_file_cstr, int line, const std::string& args)
{
	std::stringstream stream;
	stream << "Assertion failed: " << expr << " in " << extract_file_name(full_file_cstr) << ":" << line << ". "
	       << args;
	throw std::runtime_error(stream.str());
}


}
