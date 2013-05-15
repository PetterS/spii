// Petter Strandmark 2013.
//
// Test functions from
// Neculai Andrei, An Unconstrained Optimization Test Functions Collection,
// Advanced Modeling and Optimization, Volume 10, Number 1, 2008.
//

#define BEGIN_MODEL1(Model) \
	struct Model \
	{ \
		template<typename R>  \
		R operator()(const R* const x) const \
		{ \
			R value = 0;

#define BEGIN_MODEL2(Model) \
	struct Model \
	{ \
		template<typename R>  \
		R operator()(const R* const x, const R* const y) const \
		{ \
			R value = 0;

#define END_MODEL \
			return value; \
		} \
	}; 

#define LARGE_SUITE_BEGIN(Model) \
	TEST_CASE(#Model, "") \
	{ \
		auto create_function = [](std::vector<double>& start, Function* f) -> void \
		{ \
			int n = start.size();
#define LARGE_SUITE_MIDDLE \
		}; \
		auto start_value = [](int n) -> std::vector<double> \
		{ \
			std::vector<double> value(n);
#define LARGE_SUITE_END(test_newton) \
			return value; \
		}; \
		run_test(create_function, start_value, test_newton); \
	}

const bool all_methods = true;
const bool no_newton = false;

BEGIN_MODEL1(FreudenSteinRoth)
	R d1 = R(-13) + x[0] + ((R(5) - x[1])*x[1] - 2)*x[1];
	R d2 = R(-29) + x[0] + ((x[1] + R(1))*x[1] - R(14))*x[1];
	value += d1*d1 + d2*d2;
END_MODEL

LARGE_SUITE_BEGIN(FreudenSteinRoth)
	for (int i = 0; i < n/2; ++i) {
		f->add_variable(&start[2*i], 2);
	}
	for (int i = 0; i < n/2; ++i) {
		f->add_term(new AutoDiffTerm<FreudenSteinRoth, 2>(
			new FreudenSteinRoth), &start[2*i]);
	}
LARGE_SUITE_MIDDLE
	for (int i = 0; i < n; ++i) {
		if (i%2 == 0) {
			value[i] = 0.5;
		}
		else {
			value[i]  = -2.0;
		}
	}
LARGE_SUITE_END(all_methods)

// Not possible to define the Trigonometric function without a
// custom Term.

BEGIN_MODEL1(Rosenbrock)
	R d1 = 10 * (x[1] - x[0]*x[0]);
	R d2 = 1 - x[0];
	value += d1*d1 + d2*d2;
END_MODEL

LARGE_SUITE_BEGIN(Rosenbrock)
	for (int i = 0; i < n/2; ++i) {
		f->add_variable(&start[2*i], 2);
	}
	for (int i = 0; i < n/2; ++i) {
		f->add_term(new AutoDiffTerm<Rosenbrock, 2>(
			new Rosenbrock), &start[2*i]);
	}
LARGE_SUITE_MIDDLE
	for (int i = 0; i < n; ++i) {
		if (i%2 == 0) {
			value[i] = -1.2;
		}
		else {
			value[i]  = 1.0;
		}
	}
LARGE_SUITE_END(all_methods)

BEGIN_MODEL1(WhiteHolst)
	R d1 = 10 * (x[1] - x[0]*x[0]*x[0]);
	R d2 = 1 - x[0];
	value += d1*d1 + d2*d2;
END_MODEL

LARGE_SUITE_BEGIN(WhiteHolst)
	for (int i = 0; i < n/2; ++i) {
		f->add_variable(&start[2*i], 2);
	}
	for (int i = 0; i < n/2; ++i) {
		f->add_term(new AutoDiffTerm<WhiteHolst, 2>(
			new WhiteHolst), &start[2*i]);
	}
LARGE_SUITE_MIDDLE
	for (int i = 0; i < n; ++i) {
		if (i%2 == 0) {
			value[i] = -1.2;
		}
		else {
			value[i]  = 1.0;
		}
	}
LARGE_SUITE_END(all_methods)

// ...

/*
BEGIN_MODEL2(FLETCBV3a)
	value = 0.5 * (x[0]*x[0] + y[0]*y[0]);
END_MODEL

BEGIN_MODEL2(FLETCBV3b)
	R d = x[0] - y[0];
	value = 0.5 * d*d;
END_MODEL

struct FLETCBV3c 
{ 
	double h;
	FLETCBV3c(double _h) : h(_h) { }
	template<typename R> 
	R operator()(const R* const x) const
	{
		R value = 0;
		value -= (h*h + 2.0) / (h*h) * x[0] + 1.0 / (h*h) * cos(x[0]);
END_MODEL

LARGE_SUITE_BEGIN(FLETCBV3)
	for (int i = 0; i < n; ++i) {
		f.add_variable(&start[i], 1);
	}
	f.add_term(new AutoDiffTerm<FLETCBV3a, 1, 1>(
		new FLETCBV3a), &start[0], &start[n-1]);

	for (int i = 0; i < n-1; ++i) {
		f.add_term(new AutoDiffTerm<FLETCBV3b, 1, 1>(
			new FLETCBV3b), &start[i], &start[i+1]);
	}

	for (int i = 0; i < n; ++i) {
		f.add_term(new AutoDiffTerm<FLETCBV3c, 1>(
			new FLETCBV3c(1.0)), &start[i]);
	}
LARGE_SUITE_MIDDLE
	const double h = 1.0 / (n + 1.0);
	for (int i = 0; i < n; ++i) {
		value[i] = (i+1) * h;
	}
LARGE_SUITE_END(all_methods)
*/

BEGIN_MODEL1(TRIDIA1)
	R d = x[0] - 1.0;
	value = d*d;
END_MODEL

struct TRIDIA2 
{ 
	double i;
	TRIDIA2(double _i) : i(_i) { }
	template<typename R> 
	R operator()(const R* const x, const R* const y) const
	{
		R value = 0;
		R d = 2.0 * y[0] - x[0];
		value = i * d*d;
END_MODEL

LARGE_SUITE_BEGIN(TRIDIA)
	for (int i = 0; i < n; ++i) {
		f->add_variable(&start[i], 1);
	}
	f->add_term(new AutoDiffTerm<TRIDIA1, 1>(
		new TRIDIA1), &start[0]);

	for (int i = 1; i < n; ++i) {
		f->add_term(new AutoDiffTerm<TRIDIA2, 1, 1>(
			new TRIDIA2(i + 1)), &start[i-1], &start[i]);
	}

LARGE_SUITE_MIDDLE
	for (int i = 0; i < n; ++i) {
		value[i] = 1.0;
	}
LARGE_SUITE_END(all_methods)
