// Petter Strandmark 2013.
#ifndef SPII_STRING_UTILS_H
#define SPII_STRING_UTILS_H

#include <ostream>
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

	void add_to_stream(std::ostream*)
	{  }

	template<typename T, typename... Args>
	void add_to_stream(std::ostream* stream, T&& t, Args&&... args)
	{
		(*stream) << std::forward<T>(t);
		add_to_stream(stream, std::forward<Args>(args)...);
	}

	std::string to_string()
	{
		return{};
	}

	// Overload for string literals.
	template<size_t n>
	std::string to_string(const char(&c_str)[n])
	{
		return{c_str};
	}

	template<typename... Args>
	std::string to_string(Args&&... args)
	{
		std::ostringstream stream;
		add_to_stream(&stream, std::forward<Args>(args)...);
		return stream.str();
	}
}


namespace
{
	std::ostream& format_string_internal(std::ostream& stream,
	                                     const char* str,
	                                     const std::vector<std::string>& arguments)
	{
		while (*str) {
			if (*str == '%') {
				++str;
				if (*(str) == '%') {
					// OK. This will result in "%".
					stream << '%';
					++str;
				}
				else {
					int digit = *str - '0';
					++str;
					if (digit < 0 || digit > 9) {
						throw std::invalid_argument("Format specifier must be in {0, ..., 9}.");
					}
					if (digit >= arguments.size()) {
						throw std::invalid_argument("Too few arguments to format_string.");
					}

					stream << arguments.at(digit);

					// To allow format specifiers of the type "%0%".
					//if (*str == '%') {
					//	++str;
					//}
				}
			}
			else {
				stream << *str;
				++str;
			}
		}
		return stream;
	}
}


template<typename... Args>
std::ostream& format_string(std::ostream& stream, const char* str, Args&&... args)
{
	std::vector<std::string> arguments = {to_string(std::forward<Args>(args))...};
	return format_string_internal(stream, str, arguments);
}

template<typename... Args>
std::ostream& format_string(std::ostream& stream, const std::string& str, Args&&... args)
{
	return format_string(stream, str.c_str(), std::forward<Args>(args)...);
}

template<typename... Args>
std::string format_string(const char* str, Args&&... args)
{
	std::ostringstream stream;
	format_string(stream, str, std::forward<Args>(args)...);
	return stream.str();
}

template<typename... Args>
std::string format_string(const std::string& str, Args&&... args)
{
	return format_string(str.c_str(), std::forward<Args>(args)...);
}

}

#endif
