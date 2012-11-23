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
		typedef B< F<double> > BF;

		BF vars[D0];
		for (int i = 0; i < D0; ++i) {
			vars[i] = variables[0][i];
			vars[i].x().diff(i, D0); 
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
		typedef B< F<double> > BF;

		BF vars0[D0];
		for (int i = 0; i < D0; ++i) {
			vars0[i] = variables[0][i];
			vars0[i].x().diff(i, D0 + D1); 
		}
		
		BF vars1[D1];
		int offset1 = D0;
		for (int i = 0; i < D1; ++i) {
			vars1[i] = variables[1][i];
			vars1[i].x().diff(offset1 + i, D0 + D1); 
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


#endif SPII_TERM_H
