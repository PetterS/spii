// Petter Strandmark 2012.

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <spii/auto_diff_term.h>
#include <spii/term.h>

using namespace fadbad;
using namespace spii;

class Func
{
	public:
	template <class T>
	T operator()(const T* x) const
	{
		T z=sqrt(x[0]);
		return x[1]*z+sin(z);
	}
};

class DFunc
{
	public:
	template <class T>
	T operator()(
		T& o_dfdx, T& o_dfdy,
		const T& i_x, const T& i_y)
	{
		F<T, 2> x(i_x), y(i_y);
		x.diff(0);
		y.diff(1);
		Func func;
		F<T, 2> f(func(x,y));
		o_dfdx = f.d(0);
		o_dfdy = f.d(1);

		return f.x();       // Return function value
	}
};

template<typename Functor>
class DDFunc
{
	public:
	template <class T>
	T operator()(
		T& o_dfdxdx, T& o_dfdxdy,
		T& o_dfdydx, T& o_dfdydy,
		T& o_dfdx, T& o_dfdy,
		const T& i_x, const T& i_y)
	{
		Functor func;
		F<T, 2> x[2]  = {i_x, i_y};
		x[0].diff(0);
		x[1].diff(1);
		F<T, 2> df[2];
		F<T, 2> f(differentiate_functor<Functor, F<T, 2>, 2>(func, x, df));   // Evaluate function and derivatives
		o_dfdx = df[0].x();
		o_dfdy = df[1].x();
		o_dfdxdx = df[0].d(0);
		o_dfdxdy = df[0].d(1);
		o_dfdydx = df[1].d(0);
		o_dfdydy = df[1].d(1);

		return f.x();
	}

};

TEST_CASE("FADBAD/differentiate_functor", "")
{
	using namespace std;
	double f,dfdx,dfdy,
	       dfdxdx,dfdxdy,
	       dfdydx,dfdydy;
	double x=1.3;
	double y=2;
	f=DDFunc<Func>()(dfdxdx,dfdxdy,
	                 dfdydx,dfdydy,
	                 dfdx,dfdy,x,y);  // Evaluate function and derivatives

	// Check all derivatives.
	CHECK(f == y * sqrt(x) + sin(sqrt(x)));
	CHECK(dfdx == (y + cos(sqrt(x))) / (2.0*sqrt(x)));
	CHECK(dfdy == sqrt(x));
	CHECK(Approx(dfdxdx) == -(y + cos(sqrt(x)) + sqrt(x)*sin(sqrt(x)))/(4*pow(x,3.0/2.0)));
	CHECK(dfdxdy == 1.0 / (2.0*sqrt(x)));
	CHECK(dfdydx == 1.0 / (2.0*sqrt(x)));
	CHECK(dfdydy == 0.0);
}

class MyTerm: public SizedTerm<2, 3>
{
public:
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
};

TEST_CASE("SizedTerm/number_of_variables", "")
{
	MyTerm term;
	CHECK(term.number_of_variables() == 2);
}

TEST_CASE("SizedTerm/variable_dimension", "")
{
	MyTerm term;
	CHECK(term.variable_dimension(0) == 2);
	CHECK(term.variable_dimension(1) == 3);
}

class DestructorFunctor
{
public:
	DestructorFunctor(int* counter)
	{
		this->counter = counter;
	}

	~DestructorFunctor()
	{
		(*counter)++;
	}

	template<typename R>
	R operator()(const R* const x) const
	{
		return 0.0;
	}

private:
	int* counter;
};

TEST_CASE("AutoDiffTerm/calls_functor_destructor", "")
{
	int counter = 0;
	Term* term = new AutoDiffTerm<DestructorFunctor, 1>
	                 (new DestructorFunctor(&counter));
	delete term;
	CHECK(counter == 1);
}

class MyFunctor1
{
public:
	template<typename R>
	R operator()(const R* const x) const
	{
		return sin(x[0]) + cos(x[1]) + R(1.4)*x[0]*x[1] + R(1.0);
	}
};

TEST_CASE("AutoDiffTerm/MyFunctor1", "")
{
	AutoDiffTerm<MyFunctor1, 2> term(new MyFunctor1());

	double x[2] = {1.0, 3.0};
	std::vector<double*> variables;
	variables.push_back(x);

	std::vector<Eigen::VectorXd> gradient;
	gradient.push_back(Eigen::VectorXd(2));

	std::vector< std::vector<Eigen::MatrixXd> > hessian(1);
	hessian[0].resize(1);
	hessian[0][0].resize(2,2);

	double value  = term.evaluate(&variables[0], &gradient, &hessian);
	double value2 = term.evaluate(&variables[0]);

	// The two values must agree.
	CHECK(value == value2);

	// Test function value
	CHECK(value == sin(x[0]) + cos(x[1]) + 1.4*x[0]*x[1] + 1.0);

	// Test gradient
	CHECK(gradient[0](0) ==  cos(x[0]) + 1.4*x[1]);
	CHECK(gradient[0](1) == -sin(x[1]) + 1.4*x[0]);

	// Test Hessian
	CHECK(hessian[0][0](0,0) == -sin(x[0]));
	CHECK(hessian[0][0](1,1) == -cos(x[1]));
	CHECK(hessian[0][0](0,1) == 1.4);
	CHECK(hessian[0][0](1,0) == 1.4);
}

class MyFunctor2
{
public:
	template<typename R>
	R operator()(const R* const x, const R* const y) const
	{
		return sin(x[0]) + cos(y[0]) + R(1.4)*x[0]*y[0] + R(1.0);
	}
};

TEST_CASE("AutoDiffTerm/MyFunctor2", "")
{
	AutoDiffTerm<MyFunctor2, 1, 1> term(new MyFunctor2());

	double x = 5.3;
	double y = 7.1;
	std::vector<double*> variables;
	variables.push_back(&x);
	variables.push_back(&y);

	std::vector<Eigen::VectorXd> gradient;
	gradient.push_back(Eigen::VectorXd(1));
	gradient.push_back(Eigen::VectorXd(1));

	std::vector< std::vector<Eigen::MatrixXd> > hessian(2);
	hessian[0].resize(2);
	hessian[1].resize(2);
	hessian[0][0].resize(1,1);
	hessian[0][1].resize(1,1);
	hessian[1][0].resize(1,1);
	hessian[1][1].resize(1,1);

	double value  = term.evaluate(&variables[0], &gradient, &hessian);
	double value2 = term.evaluate(&variables[0]);

	// The two values must agree.
	CHECK(value == value2);

	// Test function value
	CHECK(value == sin(x) + cos(y) + 1.4*x*y + 1.0);

	// Test gradient
	CHECK(gradient[0](0) ==  cos(x) + 1.4*y);
	CHECK(gradient[1](0) == -sin(y) + 1.4*x); 

	// Test Hessian
	CHECK(hessian[0][0](0,0) == -sin(x));
	CHECK(hessian[1][1](0,0) == -cos(y));
	CHECK(hessian[1][0](0,0) == 1.4);
	CHECK(hessian[0][1](0,0) == 1.4);
}
