
#include <gtest/gtest.h>

// EXPECT_THROW gives errors on Cygwin. Disable for now.
#ifdef __CYGWIN__
	#define EXPECT_THROW(a,b)
#endif
#ifdef __GNUC__
	#define EXPECT_THROW(a,b)
#endif

#include <spii/auto_diff_term.h>
#include <spii/constraints.h>
#include <spii/function.h>

using namespace spii;

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

TEST(Function, added_same_variable_multiple_times)
{
	Function f;
	double x[5];
	f.add_variable(x, 5);
	f.add_variable(x, 5); // No-op.
	EXPECT_EQ(f.get_number_of_scalars(), 5);
	EXPECT_THROW(f.add_variable(x, 4), std::runtime_error);
}

class Term1
{
public:
	template<typename R>
	R operator()(const R* const x) const
	{
		return sin(x[0]) + cos(x[1]) + R(1.4)*x[0]*x[1] + R(1.0);
	}
};

class Term2
{
public:
	template<typename R>
	R operator()(const R* const x, const R* const y) const
	{
		return log(x[0]) + 3.0 * log(y[0]);
	}
};

TEST(Function, variable_not_found)
{
	Function f;
	double x[5];
	EXPECT_THROW(f.add_term(new AutoDiffTerm<Term1, 5>(new Term1), x), std::runtime_error);
}

TEST(Function, term_variable_mismatch)
{
	Function f;
	double x[5];
	f.add_variable(x, 5);
	EXPECT_THROW(f.add_term(new AutoDiffTerm<Term1, 4>(new Term1), x), std::runtime_error);
}

class DestructorTerm :
	public SizedTerm<1>
{
public:
	DestructorTerm(int* counter)
	{
		this->counter = counter;
	}

	~DestructorTerm()
	{
		(*counter)++;
	}

	virtual double evaluate(double * const * const variables) const
	{
		return 0;
	}

	virtual double evaluate(double * const * const variables,
	                        std::vector<Eigen::VectorXd>* gradient) const
	{
		return 0;
	}

	virtual double evaluate(double * const * const variables,
	                        std::vector<Eigen::VectorXd>* gradient,
	                        std::vector< std::vector<Eigen::MatrixXd> >* hessian) const
	{
		return 0;
	}

private:
	int* counter;
};

TEST(Function, calls_term_destructor)
{
	Function* function = new Function;
	double x[1];
	function->add_variable(x, 1);

	int counter1 = 0;
	DestructorTerm* term1 = new DestructorTerm(&counter1);
	int counter2 = 0;
	DestructorTerm* term2 = new DestructorTerm(&counter2);

	function->add_term(term1, x);
	function->add_term(term1, x);
	function->add_term(term2, x);

	EXPECT_EQ(counter1, 0);
	EXPECT_EQ(counter2, 0);
	delete function;
	EXPECT_EQ(counter1, 1);
	EXPECT_EQ(counter2, 1);

	Function* function2 = new Function;
	function->add_variable(x, 1);

	int counter3 = 0;
	DestructorTerm* term3 = new DestructorTerm(&counter3);

	function->add_term(term3, x);
	function2->term_deletion = Function::DoNotDeleteTerms;

	EXPECT_EQ(counter3, 0);
	delete function2;
	EXPECT_EQ(counter3, 0);
	delete term3;
	EXPECT_EQ(counter3, 1);
}

class DestructorChange
{
public:
	DestructorChange(int* counter)
	{
		this->counter = counter;
	}

	~DestructorChange()
	{
		(*counter)++;
	}

	template<typename R>
	void t_to_x(R* x, const R* t) const { }

	template<typename R>
	void x_to_t(R* t, const R* x) const { }

	int x_dimension() const
	{
		return 1;
	}

	int t_dimension() const
	{
		return 1;
	}
private:
	int* counter;
};

TEST(Function, calls_variable_change_destructor)
{
	Function* function = new Function;
	double x[1];
	int counter = 0;
	function->add_variable(x, 1, new DestructorChange(&counter));

	EXPECT_EQ(counter, 0);
	delete function;
	EXPECT_EQ(counter, 1);
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

	f.add_term(new AutoDiffTerm<Term1, 2>(new Term1), x);
	f.add_term(new AutoDiffTerm<Term2, 1, 1>(new Term2), y, z);

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

	f.add_term(new AutoDiffTerm<Term1, 2>(new Term1), x);
	f.add_term(new AutoDiffTerm<Term2, 1, 1>(new Term2), y, z);

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

	f.add_term(new AutoDiffTerm<Term1, 2>(new Term1), x);
	f.add_term(new AutoDiffTerm<Term1, 2>(new Term1), x);  // Add term twice for testing.
	f.add_term(new AutoDiffTerm<Term2, 1, 1>(new Term2), y, z);

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
	R operator()(const R* const x) const
	{
		return 123.4 * x[0]*x[0] + 7.0 * sin(x[1]) + 2.0 * x[0]*x[1] + 3.0 * x[2]*x[2];
	}
};

class Single2
{
public:
	template<typename R>
	R operator()(const R* const x) const
	{
		return 5.0 * x[0]*x[0] + 6.0 * x[0]*x[1] + 7.0 * x[1]*x[1];
	}
};

class Mixed3_2
{
public:
	template<typename R>
	R operator()(const R* const x, const R* const y) const
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

	f.add_term(new AutoDiffTerm<Single3, 3>(new Single3), x);
	f.add_term(new AutoDiffTerm<Single2, 2>(new Single2), y);
	f.add_term(new AutoDiffTerm<Mixed3_2, 3, 2>(new Mixed3_2), x, y);

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


TEST(Function, evaluation_count)
{

	double x[3] = {1.0, 2.0, 3.0};
	double y[2] = {3.0, 4.0};

	Function f;
	f.add_variable(x, 3);
	f.add_variable(y, 2);

	f.add_term(new AutoDiffTerm<Single3, 3>(new Single3), x);
	f.add_term(new AutoDiffTerm<Single2, 2>(new Single2), y);
	f.add_term(new AutoDiffTerm<Mixed3_2, 3, 2>(new Mixed3_2), x, y);

	Eigen::VectorXd xg(4);
	xg.setZero();
	Eigen::VectorXd gradient;
	Eigen::MatrixXd hessian;
	Eigen::SparseMatrix<double> sparse_hessian;
	f.create_sparse_hessian(&sparse_hessian);

	f.evaluate();
	f.evaluate();
	f.evaluate(xg);
	f.evaluate(xg, &gradient, &hessian);

	EXPECT_EQ(f.evaluations_without_gradient, 3);
	EXPECT_EQ(f.evaluations_with_gradient, 1);
	f.evaluate(xg, &gradient, &hessian);
	f.evaluate(xg, &gradient, &sparse_hessian);
	EXPECT_EQ(f.evaluations_with_gradient, 3);
}

//
//	x_i = exp(t_i)
//  t_i = log(x_i)
//
template<int dimension>
class ExpTransform
{
public:
	template<typename R>
	void t_to_x(R* x, const R* t) const
	{
		using std::exp;

		for (size_t i = 0; i < dimension; ++i) {
			x[i] = exp(t[i]);
		}
	}

	template<typename R>
	void x_to_t(R* t, const R* x) const
	{
		using std::log;

		for (size_t i = 0; i < dimension; ++i) {
			t[i] = log(x[i]);
		}
	}

	int x_dimension() const
	{
		return dimension;
	}

	int t_dimension() const
	{
		return dimension;
	}
};

TEST(Function, Parametrization_2_to_2)
{
	Function f1, f2;
	double x[2];
	f1.add_variable(x, 2);
	f2.add_variable(x, 2, new ExpTransform<2>);
	f1.add_term(new AutoDiffTerm<Term1, 2>(new Term1), x);
	f2.add_term(new AutoDiffTerm<Term1, 2>(new Term1), x);

	EXPECT_EQ(f1.get_number_of_scalars(), 2);
	EXPECT_EQ(f2.get_number_of_scalars(), 2);

	for (x[0] = 0.1; x[0] <= 10.0; x[0] += 0.1) {
		x[1] = x[0] / 2.0 + 0.1;
		EXPECT_NEAR(f1.evaluate(), f2.evaluate(), 1e-12);
	}

	// Term1 is
	// f(x1, x2) = sin(x1) + cos(x2) + 1.4 * x1*x2 + 1.0
	//
	// f(t1, t2) = sin(exp(t1)) + cos(exp(t2)) + 1.4 * exp(t1)*exp(t2) + 1.0
	Eigen::VectorXd x_vec(2);
	x_vec[0] = 3.0;
	x_vec[1] = 4.0;
	Eigen::VectorXd t(2);
	t[0] = std::log(x_vec[0]);
	t[1] = std::log(x_vec[1]);
	Eigen::VectorXd x_gradient;
	Eigen::VectorXd t_gradient;
	double f1_val = f1.evaluate(x_vec, &x_gradient);
	double f2_val = f2.evaluate(t, &t_gradient);

	// The function values must match.
	EXPECT_NEAR(f1_val, f2_val, 1e-12);

	// The gradient of f1 is straight-forward.
	EXPECT_NEAR(x_gradient[0],  cos(x_vec[0]) + 1.4 * x_vec[1], 1e-12);
	EXPECT_NEAR(x_gradient[1], -sin(x_vec[1]) + 1.4 * x_vec[0], 1e-12);

	// The gradient of f2 is in the transformed space.
	EXPECT_NEAR(t_gradient[0],  cos(exp(t[0]))*exp(t[0]) + 1.4 * exp(t[0]) * exp(t[1]), 1e-12);
	EXPECT_NEAR(t_gradient[1], -sin(exp(t[1]))*exp(t[1]) + 1.4 * exp(t[0]) * exp(t[1]), 1e-12);
}

TEST(Function, Parametrization_1_1_to_1_1)
{
	Function f1, f2;
	double x[1];
	double y[1];
	f1.add_variable(x, 1);
	f1.add_variable(y, 1);
	f2.add_variable(x, 1, new ExpTransform<1>);
	f2.add_variable(y, 1, new ExpTransform<1>);
	f1.add_term(new AutoDiffTerm<Term2, 1, 1>(new Term2), x, y);
	f2.add_term(new AutoDiffTerm<Term2, 1, 1>(new Term2), x, y);

	EXPECT_EQ(f1.get_number_of_scalars(), 2);
	EXPECT_EQ(f2.get_number_of_scalars(), 2);

	for (x[0] = 0.1; x[0] <= 10.0; x[0] += 0.1) {
		y[0] = x[0] / 2.0 + 0.1;
		EXPECT_NEAR(f1.evaluate(), f2.evaluate(), 1e-12);
	}

	// Term2 is
	// f(x, y) = log(x) + 3.0 * log(y);
	//
	// f(s, t) = s + 3.0 * t
	Eigen::VectorXd xy(2);
	xy[0] = 3.0;
	xy[1] = 4.0;
	Eigen::VectorXd st(2);
	st[0] = std::log(xy[0]);
	st[1] = std::log(xy[1]);
	Eigen::VectorXd xy_gradient;
	Eigen::VectorXd st_gradient;
	double f1_val = f1.evaluate(xy, &xy_gradient);
	double f2_val = f2.evaluate(st, &st_gradient);

	// The function values must match.
	EXPECT_NEAR(f1_val, f2_val, 1e-12);

	// The gradient of f1 is straight-forward.
	EXPECT_NEAR(xy_gradient[0], 1.0 / xy[0], 1e-12);
	EXPECT_NEAR(xy_gradient[1], 3.0 / xy[1], 1e-12);

	// The gradient of f2 is in the transformed space.
	EXPECT_NEAR(st_gradient[0],  1.0, 1e-12);
	EXPECT_NEAR(st_gradient[1],  3.0, 1e-12);
}

class Circle
{
public:
	template<typename R>
	void t_to_x(R* x, const R* t) const
	{
		using std::cos;
		using std::cin;

		x[0] = cos(t[0]);
		x[1] = sin(t[0]);
	}

	template<typename R>
	void x_to_t(R* t, const R* x) const
	{
		using std::atan2;

		t[0] = atan2(x[1], x[0]);
	}

	int x_dimension() const
	{
		return 2;
	}

	int t_dimension() const
	{
		return 1;
	}
};

TEST(Function, Parametrization_2_to_1)
{
	Function f1, f2;
	double x[2];
	f1.add_variable(x, 2);
	f2.add_variable(x, 2, new Circle);
	f1.add_term(new AutoDiffTerm<Term1, 2>(new Term1), x);
	f2.add_term(new AutoDiffTerm<Term1, 2>(new Term1), x);

	EXPECT_EQ(f1.get_number_of_scalars(), 2);
	EXPECT_EQ(f2.get_number_of_scalars(), 1);

	for (double theta = 0.0; theta <= 6.0; theta += 0.5) {
		x[0] = std::cos(theta);
		x[1] = std::sin(theta);
		EXPECT_NEAR(f1.evaluate(), f2.evaluate(), 1e-12);
	}
}
