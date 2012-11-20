
#include <gtest/gtest.h>

#include <spii/term.h>

class MyTerm: public SizedTerm<2, 3>
{
public:
	virtual double Evaluate(double * const * const variables) const
	{
		return 0;
	}

	virtual bool Gradient(double * const * const variables,
		                  Eigen::VectorXd* gradient)
	{
		return false;
	}

	virtual bool Hessian(double * const * const variables, Eigen::MatrixXd* hessian)
	{
		return false;
	}
};

TEST(MyTerm, number_of_variables)
{
	MyTerm term;
	EXPECT_EQ(term.number_of_variables(), 2);
}

TEST(MyTerm, variable_dimension)
{
	MyTerm term;
	EXPECT_EQ(term.variable_dimension(0), 2);
	EXPECT_EQ(term.variable_dimension(1), 3);
}
