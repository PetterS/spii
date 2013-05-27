// Petter Strandmark 2013.
#include <map>
#include <stdexcept>
#include <vector>

#include <spii/function_serializer.h>

using namespace std;

namespace spii
{

std::ostream& operator << (std::ostream& out, const Serialize& serializer)
{
	if ( ! serializer.readonly_function) {
		throw runtime_error("Serializer << : Invalid function pointer.");
	}
	const Function& f = *serializer.readonly_function;
	f.write_to_stream(out);
	return out;
}

std::istream& operator >> (std::istream& in,  const Serialize& serializer)
{
	if ( ! serializer.writable_function) {
		throw runtime_error("Serializer << : Invalid function pointer.");
	}
	if ( ! serializer.user_space) {
		throw runtime_error("Serializer << : Invalid user space pointer.");
	}
	if ( ! serializer.factory) {
		throw runtime_error("Serializer << : Invalid factory pointer.");
	}
	Function& f = *serializer.writable_function;
	f.read_from_stream(in, serializer.user_space, *serializer.factory);
	return in;
}

}
