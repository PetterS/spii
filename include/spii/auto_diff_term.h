#ifndef SPII_AUTO_DIFF_TERM_H
#define SPII_AUTO_DIFF_TERM_H

#include <badiff.h>
#include <fadiff.h>

#include <spii/term.h>

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

//
// 1-variable specialization
//
template<typename Functor, int D0> 
class AutoDiffTerm<Functor, D0, 0, 0, 0> :
	public SizedTerm<D0, 0, 0, 0>
{
public:
	AutoDiffTerm(Functor* f)
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
	                        std::vector<Eigen::VectorXd>* gradient,
	                        std::vector< std::vector<Eigen::MatrixXd> >* hessian) const
	{
		using namespace fadbad;
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
	}

private:
	Functor* functor;
};


//
// 2-variable specialization
//
template<typename Functor, int D0, int D1> 
class AutoDiffTerm<Functor, D0, D1, 0, 0> :
	public SizedTerm<D0, D1, 0, 0>
{
public:
	AutoDiffTerm(Functor* f)
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
	                        std::vector<Eigen::VectorXd>* gradient,
	                        std::vector< std::vector<Eigen::MatrixXd> >* hessian) const
	{
		using namespace fadbad;
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
	}

private:
	Functor* functor;
};


#endif
