// Petter Strandmark 2013.
#ifndef SPII_TERM_FACTORY_H
#define SPII_TERM_FACTORY_H

#include <functional>
#include <iostream>
#include <string>
#include <typeinfo>

//#include <spii/auto_diff_term.h>
#include <spii/spii.h>
#include <spii/term.h>

namespace spii
{

class SPII_API TermFactory
{
public:
	typedef std::function<spii::Term*(std::istream&)> TermCreator;

	TermFactory();

	template<typename T>
	void teach_term()
	{
		TermTeacher<T>::teach(*this);
	}

	static std::string fix_name(const std::string& org_name);

	Term* create(const std::string& term_name, std::istream& in) const;
	void teach_term(const std::string& term_name, const TermCreator& creator);

private:
	class Implementation;
	// unique_pointer would have been nice, but there are issues
	// with sharing these objects across DLL boundaries in VC++.
	Implementation* impl;
};

//
// The TermTeacher class is used because partially specializing
// functions is not possible.
//
template<typename T>
struct TermTeacher
{
	static void teach(TermFactory& factory)
	{
		auto creator = [](std::istream& in) -> spii::Term*
		{
			auto term = new T;
			in >> *term;
			return new T;
		};
		factory.teach_term(factory.fix_name(typeid(T).name()), creator);
	}
};

// Specialize for AutoDiffTerm.
template<typename Functor, int D0, int D1 = 0, int D2 = 0, int D3 = 0>
class AutoDiffTerm;
template<typename T, int D0, int D1, int D2, int D3>
struct TermTeacher< AutoDiffTerm<T,D0,D1,D2,D3> >
{
	typedef AutoDiffTerm<T,D0,D1,D2,D3> AutoTerm;
	static void teach(TermFactory& factory)
	{
		auto creator = [](std::istream& in) -> spii::Term*
		{
			auto term = new AutoTerm(new T);
			in >> *term;
			return term;
		};
		factory.teach_term(typeid(AutoTerm).name(), creator);
	}
};

}

#endif
