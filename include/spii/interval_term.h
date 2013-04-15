#ifndef SPII_INTERVAL_TERM
#define SPII_INTERVAL_TERM

#include <spii/auto_diff_term.h>

namespace spii
{

template<typename Functor, int D0, int D1 = 0, int D2 = 0, int D3 = 0>
class IntervalTerm :
	public AutoDiffTerm<Functor, D0, D1, D2, D3>
{

};

//
// 1-variable specialization
//
template<typename Functor, int D0>
class IntervalTerm<Functor, D0, 0, 0, 0> :
	public AutoDiffTerm<Functor, D0, 0, 0, 0>
{
public:
	IntervalTerm(const Functor* f):
		AutoDiffTerm<Functor, D0, 0, 0, 0>(f)
	{
	}

	virtual Interval<double> evaluate_interval(const Interval<double> * const * const variables) const
	{
		return (*this->functor)(variables[0]);
	};
};

//
// 2-variable specialization
//
template<typename Functor, int D0, int D1>
class IntervalTerm<Functor, D0, D1, 0, 0> :
	public AutoDiffTerm<Functor, D0, D1, 0, 0>
{
public:
	IntervalTerm(const Functor* f):
		AutoDiffTerm<Functor, D0, D1, 0, 0>(f)
	{
	}

	virtual Interval<double> evaluate_interval(const Interval<double> * const * const variables) const
	{
		return (*this->functor)(variables[0], variables[1]);
	};
};

}  // namespace spii
#endif
