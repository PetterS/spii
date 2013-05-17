// Petter Strandmark 2012.
#ifndef SPII_TERM_H
#define SPII_TERM_H
// The Term class defines a single term in an objective function.
// Ususally a term is created via the AutoDiffTerm template class
// so that the derivatives do not have to be explicitly computed.

#include <cstddef>
#include <stdexcept>
#include <vector>
using std::size_t;

#include <Eigen/Core>

#include <spii/interval.h>

namespace spii
{

class Term
{
public:
	virtual ~Term() {};
	virtual int number_of_variables() const       = 0;
	virtual int variable_dimension(int var) const = 0;
	virtual double evaluate(double * const * const variables) const = 0;
	virtual double evaluate(double * const * const variables,
	                        std::vector<Eigen::VectorXd>* gradient) const = 0;
	virtual double evaluate(double * const * const variables,
	                        std::vector<Eigen::VectorXd>* gradient,
	                        std::vector< std::vector<Eigen::MatrixXd> >* hessian) const = 0;

	// This function only needs to be implemented if interval arithmetic is
	// desired.
	virtual Interval<double> evaluate_interval(const Interval<double> * const * const variables) const
	{
		throw std::runtime_error("evaluate_interval: Not implemented.");
	};
};

template<int D0,int D1 = 0, int D2 = 0, int D3 = 0>
class SizedTerm :
	public Term
{
public:
	virtual int number_of_variables() const
	{
		if (D1 == 0) return 1;
		if (D2 == 0) return 2;
		if (D3 == 0) return 3;
		return 4;
	}

	virtual int variable_dimension(int var) const
	{
		if (var == 0) return D0;
		if (var == 1) return D1;
		if (var == 2) return D2;
		if (var == 3) return D3;
		return -1;
	}
};

}  // namespace spii
#endif
