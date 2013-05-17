#ifndef SPII_AUTO_DIFF_TERM_H
#define SPII_AUTO_DIFF_TERM_H

#include <spii-thirdparty/badiff.h>
#include <spii-thirdparty/fadiff.h>

#include <spii/term.h>

namespace spii {

//
// Term which allows for automatic computation of derivatives. It is
// used in the following way:
//
//   new AutoDiffTerm<Functor, 1>( new Functor(...) )
//
// Note that AutoDiffTerm always takes ownership of the functor passed
// to the constructor. It will delete it when its destructor is called.
//
template<typename Functor, int D0, int D1 = 0, int D2 = 0, int D3 = 0>
class AutoDiffTerm :
	public SizedTerm<D0, D1, D2, D3>
{

};

// to_double(x) returns the real part of x, disregarding
// any derivatives.
inline double to_double(double x)
{
	return x;
}
template<typename R>
inline double to_double(R& x)
{
	return to_double(x.x());
}

// Function differentiating a functor taking D variables.
template<typename Functor, typename T, int D>
T differentiate_functor(
	const Functor& functor,
	const T* x_in,
	T* df)
{
	using namespace fadbad;

	F<T, D> x[D];
	for (int i=0; i<D; ++i) {
		x[i] = x_in[i];
		x[i].diff(i);
	}
	F<T, D> f(functor(x));

	for (int i=0; i<D; ++i) {
		df[i] = f.d(i);
	}

	return f.x();
}

//
// 1-variable specialization
//
template<typename Functor, int D0>
class AutoDiffTerm<Functor, D0, 0, 0, 0> :
	public SizedTerm<D0, 0, 0, 0>
{
public:
	AutoDiffTerm(const Functor* f)
	{
		this->functor = f;
	}

	~AutoDiffTerm()
	{
		delete this->functor;
	}

	virtual double evaluate(double * const * const variables) const
	{
		return (*functor)(variables[0]);
	}

	virtual double evaluate(double * const * const variables,
	                        std::vector<Eigen::VectorXd>* gradient) const
	{
		using namespace fadbad;

		F<double, D0> vars[D0];
		for (int i = 0; i < D0; ++i) {
			vars[i] = variables[0][i];
			vars[i].diff(i);
		}

		F<double, D0> f((*functor)(vars));

		for (int i = 0; i < D0; ++i) {
			(*gradient)[0](i) = f.d(i);
		}

		return f.x();
	}

	virtual double evaluate(double * const * const variables,
	                        std::vector<Eigen::VectorXd>* gradient,
	                        std::vector< std::vector<Eigen::MatrixXd> >* hessian) const
	{
		using namespace fadbad;
		#ifdef USE_BF_DIFFERENTIATION
			typedef B< F<double, D0> > BF;

			BF vars[D0];
			for (int i = 0; i < D0; ++i) {
				vars[i] = variables[0][i];
				vars[i].x().diff(i);
			}

			BF f = (*functor)(vars);
			f.diff(0, 1);

			for (int i = 0; i < D0; ++i) {
				(*gradient)[0](i) = vars[i].d(0).x();
				for (int j = 0; j < D0; ++j) {
					(*hessian)[0][0](i, j) = vars[i].d(0).d(j);
				}
			}

			return f.x().x();
		#else
			F<double, D0> vars[D0];
			for (int i = 0; i < D0; ++i) {
				vars[i] = variables[0][i];
				vars[i].diff(i);
			}

			F<double, D0> df[D0];

			F<double, D0> f(
				differentiate_functor<Functor, F<double, D0>, D0>(
					*functor,
					vars,
					df)
				);

			for (int i = 0; i < D0; ++i) {
				(*gradient)[0](i) = df[i].x();
				for (int j = 0; j < D0; ++j) {
					(*hessian)[0][0](i, j) = df[i].d(j);
				}
			}

			return f.x();
		#endif
	}

protected:
	const Functor* functor;
};


template<typename Functor, int D0, int D1>
class Functor2_to_1
{
public:
	Functor2_to_1(const Functor* functor)
	{
		this->functor = functor;
	}

	template<typename R>
	R operator()(const R* const x) const
	{
		const R* const x0 = &x[0];
		const R* const x1 = &x[D0];
		return (*functor)(x0, x1);
	}

private:
	const Functor* functor;
};

//
// 2-variable specialization
//
template<typename Functor, int D0, int D1>
class AutoDiffTerm<Functor, D0, D1, 0, 0> :
	public SizedTerm<D0, D1, 0, 0>
{
public:
	AutoDiffTerm(const Functor* f)
	{
		this->functor = f;
	}

	~AutoDiffTerm()
	{
		delete this->functor;
	}

	virtual double evaluate(double * const * const variables) const
	{
		return (*functor)(variables[0], variables[1]);
	}

	virtual double evaluate(double * const * const variables,
	                        std::vector<Eigen::VectorXd>* gradient) const
	{
		using namespace fadbad;

		F<double, D0 + D1> vars0[D0];
		for (int i = 0; i < D0; ++i) {
			vars0[i] = variables[0][i];
			vars0[i].diff(i);
		}

		F<double, D0 + D1> vars1[D1];
		int offset1 = D0;
		for (int i = 0; i < D1; ++i) {
			vars1[i] = variables[1][i];
			vars1[i].diff(i + offset1);
		}

		F<double, D0 + D1> f((*functor)(vars0, vars1));

		for (int i = 0; i < D0; ++i) {
			(*gradient)[0](i) = f.d(i);
		}

		for (int i = 0; i < D1; ++i) {
			(*gradient)[1](i) = f.d(i + offset1);
		}

		return f.x();
	}

	virtual double evaluate(double * const * const variables,
	                        std::vector<Eigen::VectorXd>* gradient,
	                        std::vector< std::vector<Eigen::MatrixXd> >* hessian) const
	{
		using namespace fadbad;
		#ifdef USE_BF_DIFFERENTIATION
			typedef B< F<double, D0 + D1> > BF;

			BF vars0[D0];
			for (int i = 0; i < D0; ++i) {
				vars0[i] = variables[0][i];
				vars0[i].x().diff(i);
			}

			BF vars1[D1];
			int offset1 = D0;
			for (int i = 0; i < D1; ++i) {
				vars1[i] = variables[1][i];
				vars1[i].x().diff(offset1 + i);
			}

			BF f = (*functor)(vars0, vars1);
			f.diff(0, 1);

			for (int i = 0; i < D0; ++i) {
				(*gradient)[0](i) = vars0[i].d(0).x();

				// D0 and D0
				for (int j = 0; j < D0; ++j) {
					(*hessian)[0][0](i, j) = vars0[i].d(0).d(j);
				}

				// D0 and D1
				for (int j = 0; j < D1; ++j) {
					(*hessian)[0][1](i, j) = vars0[i].d(0).d(offset1 + j);
				}
			}

			for (int i = 0; i < D1; ++i) {
				(*gradient)[1](i) = vars1[i].d(0).x();

				// D1 and D0
				for (int j = 0; j < D0; ++j) {
					(*hessian)[1][0](i, j) = vars1[i].d(0).d(j);
				}

				// D1 and D1
				for (int j = 0; j < D1; ++j) {
					(*hessian)[1][1](i, j) = vars1[i].d(0).d(offset1 + j);
				}
			}

			return f.x().x();
		#else
			F<double, D0 + D1> vars[D0 + D1];
			F<double, D0 + D1>   df[D0 + D1];

			// Initialize variables
			for (int i = 0; i < D0; ++i) {
				vars[i] = variables[0][i];
				vars[i].diff(i);
			}
			int offset1 = D0;
			for (int i = 0; i < D1; ++i) {
				vars[offset1 + i] = variables[1][i];
				vars[offset1 + i].diff(offset1 + i);
			}

			// Evaluate function
			typedef Functor2_to_1<Functor, D0, D1> Functor21;
			Functor21 functor21(functor);
			F<double, D0 + D1> f(
				differentiate_functor<Functor21, F<double, D0 + D1>, D0 + D1>(
					functor21,
					vars,
					df)
				);

			for (int i = 0; i < D0; ++i) {
				(*gradient)[0](i) = df[i].x();

				// D0 and D0
				for (int j = 0; j < D0; ++j) {
					(*hessian)[0][0](i, j) = df[i].d(j);
				}

				// D0 and D1
				for (int j = 0; j < D1; ++j) {
					(*hessian)[0][1](i, j) = df[i].d(offset1 + j);
				}
			}

			for (int i = 0; i < D1; ++i) {
				(*gradient)[1](i) = df[i + offset1].x();;

				// D1 and D0
				for (int j = 0; j < D0; ++j) {
					(*hessian)[1][0](i, j) = df[i + offset1].d(j);;
				}

				// D1 and D1
				for (int j = 0; j < D1; ++j) {
					(*hessian)[1][1](i, j) = df[i + offset1].d(j + offset1);
				}
			}

			return f.x();
		#endif
	}

protected:
	const Functor* functor;
};

}  // namespace spii

#endif
