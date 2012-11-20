#ifndef SPII_TERM_H

#include <cstddef>
using std::size_t;

#include <Eigen/Core>

class Term
{
public:
	virtual int number_of_variables() const                          = 0;
	virtual int variable_dimension(int var) const                    = 0;
	virtual double Evaluate(double * const * const variables) const  = 0;
	virtual bool Gradient(double * const * const variables, Eigen::VectorXd* gradient) const = 0;
	virtual bool Hessian(double * const * const variables, Eigen::MatrixXd* hessian) const   = 0;
};

template<int D0,int D1 = 0, int D2 = 0, int D3 = 0> 
class SizedTerm
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


#endif SPII_TERM_H
