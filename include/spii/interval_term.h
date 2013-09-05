#ifndef SPII_INTERVAL_TERM
#define SPII_INTERVAL_TERM

#include <spii/auto_diff_term.h>

namespace spii
{

template<typename Functor, int D0, int D1 = 0, int D2 = 0, int D3 = 0, int D4 = 0>
class IntervalTerm :
	public AutoDiffTerm<Functor, D0, D1, D2, D3, D4>
{

};

//
// 1-variable specialization
//
template<typename Functor, int D0>
class IntervalTerm<Functor, D0, 0, 0, 0, 0> :
	public AutoDiffTerm<Functor, D0, 0, 0, 0, 0>
{
public:
	// When compilers (MSVC) support variadic templates, this code will
	// be shorter.
	IntervalTerm()
	{ 
	}
	template<typename T1>
	IntervalTerm(T1&& t1)
		: AutoDiffTerm<Functor, D0, 0, 0, 0, 0>(std::forward<T1>(t1))
	{ 
	}
	template<typename T1, typename T2>
	IntervalTerm(T1&& t1, T2&& t2)
		: AutoDiffTerm<Functor, D0, 0, 0, 0, 0>(std::forward<T1>(t1), std::forward<T2>(t2))
	{
	}
	template<typename T1, typename T2, typename T3>
	IntervalTerm(T1&& t1, T2&& t2, T3&& t3)
		: AutoDiffTerm<Functor, D0, 0, 0, 0, 0>(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3))
	{
	}
	// Etc. if needed.

	virtual Interval<double> evaluate_interval(const Interval<double> * const * const variables) const
	{
		return this->functor(variables[0]);
	};
};

//
// 2-variable specialization
//
template<typename Functor, int D0, int D1>
class IntervalTerm<Functor, D0, D1, 0, 0, 0> :
	public AutoDiffTerm<Functor, D0, D1, 0, 0, 0>
{
public:
	// When compilers (MSVC) support variadic templates, this code will
	// be shorter.
	IntervalTerm()
	{ 
	}
	template<typename T1>
	IntervalTerm(T1&& t1)
		: AutoDiffTerm<Functor, D0, D1, 0, 0, 0>(std::forward<T1>(t1))
	{ 
	}
	template<typename T1, typename T2>
	IntervalTerm(T1&& t1, T2&& t2)
		: AutoDiffTerm<Functor, D0, D1, 0, 0, 0>(std::forward<T1>(t1), std::forward<T2>(t2))
	{
	}
	template<typename T1, typename T2, typename T3>
	IntervalTerm(T1&& t1, T2&& t2, T3&& t3)
		: AutoDiffTerm<Functor, D0, D1, 0, 0, 0>(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3))
	{
	}
	// Etc. if needed.

	virtual Interval<double> evaluate_interval(const Interval<double> * const * const variables) const
	{
		return this->functor(variables[0], variables[1]);
	};
};

}  // namespace spii
#endif
