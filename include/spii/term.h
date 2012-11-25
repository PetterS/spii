#ifndef SPII_TERM_H
#define SPII_TERM_H

#include <cstddef>
#include <vector>
using std::size_t;

#include <Eigen/Core>

#include <badiff.h>
#include <fadiff.h>

class Term
{
public:
	virtual ~Term() {};
	virtual int number_of_variables() const       = 0;
	virtual int variable_dimension(int var) const = 0;
	virtual double evaluate(double * const * const variables) const = 0;
	virtual double evaluate(double * const * const variables,
	                        std::vector<Eigen::VectorXd>* gradient,
	                        std::vector< std::vector<Eigen::MatrixXd> >* hessian) const = 0;
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

#endif
