// Petter Strandmark 2013.
#include <map>
#include <stdexcept>
#include <vector>

#include <spii/term_factory.h>

using namespace std;

namespace spii
{

class TermFactory::Implementation
{
public:
	map<string, TermCreator> creators;
};

TermFactory::TermFactory() :
	impl(new TermFactory::Implementation)
{
}

Term* TermFactory::create(const std::string& term_name,
                          std::istream& in) const
{
	auto creator = impl->creators.find(fix_name(term_name));
	if (creator == impl->creators.end()) {
		std::string msg = "TermFactory::create: Unknown Term ";
		msg += term_name;
		throw runtime_error(msg.c_str());
	}
	return creator->second(in);
}

void TermFactory::teach_term(const std::string& term_name, const TermCreator& creator)
{
	impl->creators[fix_name(term_name)] = creator;
}

std::string TermFactory::fix_name(const std::string& org_name)
{
	std::string name = org_name;
	std::replace(std::begin(name), std::end(name), ' ', '-');
	return name;
}

}
