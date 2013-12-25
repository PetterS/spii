// Petter Strandmark 2013.
#ifndef SPII_STRING_UTILS_H
#define SPII_STRING_UTILS_H

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spii
{

// to_string converts all its arguments to a string and
// concatenates.
//
// Anonymous namespace is needed because the base case for the
// template recursion is a normal function.
namespace {

	template<typename T, typename... Args>
	void add_to_stream(std::ostream* stream, T&& t, Args&&... args)
	{
		(*stream) << std::forward<T>(t);
		add_to_stream(stream, std::forward<Args>(args)...);
	}

	void add_to_stream(std::ostream*)
	{  }

	template<typename... Args>
	std::string to_string(Args&&... args)
	{
		std::ostringstream stream;
		add_to_stream(&stream, std::forward<Args>(args)...);
		return stream.str();
	}

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
}


}

#endif
