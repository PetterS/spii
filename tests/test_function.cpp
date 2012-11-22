
#include <gtest/gtest.h>

#include <spii/function.h>

TEST(Function, get_number_of_scalars)
{
	Function f;
	double x[5];
	double y[4];
	double z[2];
	f.add_variable(x, 5);
	f.add_variable(y, 4);
	f.add_variable(z, 2);
	EXPECT_EQ(f.get_number_of_scalars(), 11);
}

TEST(Function, dimension_mismatch)
{
	Function f;
	double x[5];
	f.add_variable(x, 5);
	f.add_variable(x, 5); // No-op.
	EXPECT_THROW(f.add_variable(x, 4), std::runtime_error);
}

class Term1
{
public:
	template<typename R>
	R operator()(const R* const x)
	{
		return sin(x[0]) + cos(x[1]) + R(1.4)*x[0]*x[1] + R(1.0);
	}
};

class Term2
{
public:
	template<typename R>
	R operator()(const R* const x, const R* const y)
	{
		return log(x[0]) + 3.0 * log(y[0]);
	}
};

TEST(Function, term_variable_mismatch)
{
	Function f;
	double x[5];
	f.add_variable(x, 5);
	Term1 term1;
	EXPECT_THROW(f.add_term(new AutoDiffTerm<Term1, 4>(&term1), x), std::runtime_error);
}

TEST(Function, evaluate)
{
	
	double x[2] = {1.0, 2.0};
	double y[1] = {3.0};
	double z[1] = {4.0};

	Function f;
	f.add_variable(x, 2);
	f.add_variable(y, 1);
	f.add_variable(z, 1);

	Term1 term1;
	f.add_term(new AutoDiffTerm<Term1, 2>(&term1), x);
	Term2 term2;
	f.add_term(new AutoDiffTerm<Term2, 1, 1>(&term2), y, z);
	
	double fval = f.evaluate();
	EXPECT_DOUBLE_EQ(fval, sin(x[0]) + cos(x[1]) + 1.4 * x[0]*x[1] + 1.0 +
	                       log(y[0]) + 3.0 * log(z[0]));
}

TEST(Function, evaluate_x)
{
	
	double x[2] = {1.0, 2.0};
	double y[1] = {3.0};
	double z[1] = {4.0};

	Function f;
	f.add_variable(x, 2);
	f.add_variable(y, 1);
	f.add_variable(z, 1);

	Term1 term1;
	f.add_term(new AutoDiffTerm<Term1, 2>(&term1), x);
	Term2 term2;
	f.add_term(new AutoDiffTerm<Term2, 1, 1>(&term2), y, z);
	
	Eigen::VectorXd xg(4);
	xg[0] = 6.0;
	xg[1] = 7.0;
	xg[2] = 8.0;
	xg[3] = 9.0;

	double fval = f.evaluate(xg);
	EXPECT_DOUBLE_EQ(fval, sin(xg[0]) + cos(xg[1]) + 1.4 * xg[0]*xg[1] + 1.0 +
	                       log(xg[2]) + 3.0 * log(xg[3]));
}


TEST(Function, evaluate_gradient)
{
	
	double x[2] = {1.0, 2.0};
	double y[1] = {3.0};
	double z[1] = {4.0};

	Function f;
	f.add_variable(x, 2);
	f.add_variable(y, 1);
	f.add_variable(z, 1);

	Term1 term1;
	f.add_term(new AutoDiffTerm<Term1, 2>(&term1), x);
	f.add_term(new AutoDiffTerm<Term1, 2>(&term1), x);  // Add term twice for testing.
	Term2 term2;
	f.add_term(new AutoDiffTerm<Term2, 1, 1>(&term2), y, z);
	
	Eigen::VectorXd xg(4);
	xg[0] = 6.0;
	xg[1] = 7.0;
	xg[2] = 8.0;
	xg[3] = 9.0;

	Eigen::VectorXd gradient;
	Eigen::MatrixXd hessian;

	double fval = f.evaluate(xg, &gradient, &hessian);
	EXPECT_EQ(gradient.size(), 4);

	// Check gradient values.
	// x term was added twice, hence 2.0.
	EXPECT_DOUBLE_EQ(gradient[0], 2.0 * (cos(xg[0]) + 1.4 * xg[1]));
	EXPECT_DOUBLE_EQ(gradient[1], 2.0 * (-sin(xg[1]) + 1.4 * xg[0]));
	EXPECT_DOUBLE_EQ(gradient[2], 1.0 / xg[2]);
	EXPECT_DOUBLE_EQ(gradient[3], 3.0 / xg[3]);
}

class Single3
{
public:
	template<typename R>
	R operator()(const R* const x)
	{
		return 123.4 * x[0]*x[0] + 7.0 * sin(x[1]) + 2.0 * x[0]*x[1] + 3.0 * x[2]*x[2];
	}
};

class Single2
{
public:
	template<typename R>
	R operator()(const R* const x)
	{
		return 5.0 * x[0]*x[0] + 6.0 * x[0]*x[1] + 7.0 * x[1]*x[1];
	}
};

class Mixed3_2
{
public:
	template<typename R>
	R operator()(const R* const x, const R* const y)
	{
		return    9.0 * x[0]*y[0] + 10.0 * x[0]*y[1]
		       + 11.0 * x[1]*y[0] + 12.0 * x[1]*y[1]
			   + 13.0 * x[2]*y[0] + 14.0 * cos(x[2]*y[1]);
	}
};

TEST(Function, evaluate_hessian)
{
	
	double x[3] = {1.0, 2.0, 3.0};
	double y[2] = {3.0, 4.0};

	Function f;
	f.add_variable(x, 3);
	f.add_variable(y, 2);

	Single3 single3;
	f.add_term(new AutoDiffTerm<Single3, 3>(&single3), x);
	Single2 single2;
	f.add_term(new AutoDiffTerm<Single2, 2>(&single2), y);
	Mixed3_2 mixed3_2;
	f.add_term(new AutoDiffTerm<Mixed3_2, 3, 2>(&mixed3_2), x, y);
	
	Eigen::VectorXd xg(4);
	xg[0] = 6.0; // x[0]
	xg[1] = 7.0; // x[1]
	xg[2] = 8.0; // x[2]
	xg[3] = 9.0; // y[0]
	xg[4] = 1.0; // y[1]

	Eigen::VectorXd gradient;
	Eigen::MatrixXd hessian;

	double fval = f.evaluate(xg, &gradient, &hessian);
	ASSERT_EQ(hessian.rows(), 5);
	ASSERT_EQ(hessian.cols(), 5);

	// Check the x part of hessian.
	EXPECT_DOUBLE_EQ(hessian(0,0), 2.0 * 123.4);
	EXPECT_DOUBLE_EQ(hessian(1,1), - 7.0 * sin(xg[1]));
	EXPECT_DOUBLE_EQ(hessian(2,2), 2.0 * 3.0
	                               - 14.0 * xg[4] * xg[4] * cos(xg[2]*xg[4]));

	EXPECT_DOUBLE_EQ(hessian(0,1), 2.0);
	EXPECT_DOUBLE_EQ(hessian(1,0), 2.0);
	EXPECT_DOUBLE_EQ(hessian(0,2), 0.0);
	EXPECT_DOUBLE_EQ(hessian(2,0), 0.0);
	EXPECT_DOUBLE_EQ(hessian(1,2), 0.0);
	EXPECT_DOUBLE_EQ(hessian(2,1), 0.0);
	
	// Check the y part of hessian.
	EXPECT_DOUBLE_EQ(hessian(3,3), 2.0 * 5.0);
	EXPECT_DOUBLE_EQ(hessian(4,4), 2.0 * 7.0
		                           - 14.0 * xg[2] * xg[2] * cos(xg[2]*xg[4]));
	EXPECT_DOUBLE_EQ(hessian(3,4), 6.0);
	EXPECT_DOUBLE_EQ(hessian(4,3), 6.0);

	// Check the x-y part of hessian.
	EXPECT_DOUBLE_EQ(hessian(0,3),  9.0);
	EXPECT_DOUBLE_EQ(hessian(3,0),  9.0);
	EXPECT_DOUBLE_EQ(hessian(0,4), 10.0);
	EXPECT_DOUBLE_EQ(hessian(4,0), 10.0);
	EXPECT_DOUBLE_EQ(hessian(1,3), 11.0);
	EXPECT_DOUBLE_EQ(hessian(3,1), 11.0);
	EXPECT_DOUBLE_EQ(hessian(1,4), 12.0);
	EXPECT_DOUBLE_EQ(hessian(4,1), 12.0);
	EXPECT_DOUBLE_EQ(hessian(2,3), 13.0);
	EXPECT_DOUBLE_EQ(hessian(3,2), 13.0);
	EXPECT_DOUBLE_EQ(hessian(2,4), - 14.0 * (sin(xg[2]*xg[4]) + xg[2] * xg[4] * cos(xg[2]*xg[4])));
	EXPECT_DOUBLE_EQ(hessian(4,2), - 14.0 * (sin(xg[2]*xg[4]) + xg[2] * xg[4] * cos(xg[2]*xg[4])));
}

