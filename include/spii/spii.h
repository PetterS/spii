// Petter Strandmark 2012–2013.
#ifndef SPII_H
#define SPII_H

#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#	ifdef spii_EXPORTS
#		define SPII_API __declspec(dllexport)
#		define SPII_API_EXTERN_TEMPLATE
#	else
#		define SPII_API __declspec(dllimport)
#		define SPII_API_EXTERN_TEMPLATE extern
#	endif
#else
#	define SPII_API
#	define SPII_API_EXTERN_TEMPLATE
#endif // WIN32

namespace spii
{

double SPII_API wall_time();


// Helper functions that sends all its arguments to the provided
// pointer to stream.
//
// Anonymous namespace is needed because the base case for the
// template recursion is a normal function.
namespace {

	std::string to_string()
	{ 
		return {};
	}

	// Overload for string literals.
	template<size_t n>
	std::string to_string(const char(&c_str)[n])
	{
		return{ c_str };
	}

	template<typename T, typename... Args>
	std::string to_string(T&& t, Args&&... args)
	{
		std::stringstream stream;
		stream << std::forward<T>(t)
			   << to_string(std::forward<Args>(args)...);
		return stream.str();
	}
}


//
// Enables expressions like:
//
//    check(a == 42, a, " is not 42.");
//
// Will throw if expression is false.
//
template<typename... Args>
void check(bool everything_OK, const char* message)
{
	if (!everything_OK) {
		throw std::runtime_error(message);
	}
}

template<typename... Args>
void check(bool everything_OK, Args&&... args)
{
	if (!everything_OK) {
		throw std::runtime_error(to_string(std::forward<Args>(args)...));
	}
}

// Prepares a message and throws an exception.
void SPII_API verbose_error_internal(const char* expr, const char* full_file_cstr, int line, const std::string& args);

template<typename... Args>
void verbose_error(const char* expr, const char* full_file_cstr, int line, Args&&... args)
{
	verbose_error_internal(expr, full_file_cstr, line, to_string(std::forward<Args>(args)...));
}

#define spii_assert(expr, ...) (expr) ? ((void)0) : spii::verbose_error(#expr, __FILE__, __LINE__, spii::to_string(__VA_ARGS__))

}

#endif
