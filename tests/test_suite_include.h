// Include this function in a file defining the run_test
// template function.
//
// Petter Strandmark 2012
//
// Test functions from
// Jorge J. More, Burton S. Garbow and Kenneth E. Hillstrom,
// "Testing unconstrained optimization software",
// Transactions on Mathematical Software 7(1):17-41, 1981.
// http://www.caam.rice.edu/~zhang/caam454/nls/MGH.pdf
//
struct Rosenbrock
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R d0 =  x[1] - x[0]*x[0];
		R d1 =  1 - x[0];
		return 100 * d0*d0 + d1*d1;
	}
};

TEST(Solver, Rosenbrock)
{
	double x[2] = {-1.2, 1.0};
	double fval = run_test<Rosenbrock, 2>(x);

	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(x[1] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(fval), 1e-9);
}

TEST(Solver, RosenbrockFar)
{
	double x[2] = {-1e6, 1e6};

	Solver solver;
	create_solver(&solver);
	solver.gradient_tolerance = 1e-40;
	solver.maximum_iterations = 100000;
	double fval = run_test<Rosenbrock, 2>(x, &solver);

	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(x[1] - 1.0), 1e-9);
	EXPECT_LT( std::fabs(fval), 1e-9);
}

struct FreudenStein_Roth
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R d0 =  -13.0 + x[0] + ((5.0 - x[1])*x[1] - 2.0)*x[1];
		R d1 =  -29.0 + x[0] + ((x[1] + 1.0)*x[1] - 14.0)*x[1];
		return d0*d0 + d1*d1;
	}
};

TEST(Solver, FreudenStein_Roth)
{
	double x[2] = {0.5, -2.0};
	double fval = run_test<FreudenStein_Roth, 2>(x);

	// Can end up in local minima 48.9842...
	//EXPECT_LT( std::fabs(x[0] - 5.0), 1e-9);
	//EXPECT_LT( std::fabs(x[1] - 4.0), 1e-9);
	//EXPECT_LT( std::fabs(f.evaluate()), 1e-9);
}

struct Powell_badly_scaled
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R d0 = 1e4*x[0]*x[1] - 1;
		R d1 = exp(-x[0]) + exp(-x[1]) - 1.0001;
		return d0*d0 + d1*d1;
	}
};

TEST(Solver, Powell_badly_scaled)
{
	double x[2] = {0.0, 1.0};
	double fval = run_test<Powell_badly_scaled, 2>(x);

	EXPECT_LT( std::fabs(fval), 1e-9);
}

struct Brown_badly_scaled
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R d0 = x[0] - 1e6;
		R d1 = x[1] - 2e-6;
		R d2 = x[0]*x[1] - 2;
		return d0*d0 + d1*d1 + d2*d2;
	}
};

TEST(Solver, Brown_badly_scaled)
{
	double x[2] = {1.0, 1.0};
	double fval = run_test<Brown_badly_scaled, 2>(x);

	EXPECT_LT( std::fabs(x[0] - 1e6),  1e-3);
	EXPECT_LT( std::fabs(x[1] - 2e-6), 1e-9);
	EXPECT_LT( std::fabs(fval), 1e-9);
}


struct Beale
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R d0 = 1.5   - x[0] * (1.0 - x[1]);
		R d1 = 2.25  - x[0] * (1.0 - x[1]*x[1]);
		R d2 = 2.625 - x[0] * (1.0 - x[1]*x[1]*x[1]);
		return d0*d0 + d1*d1 + d2*d2;
	}
};

TEST(Solver, Beale)
{
	double x[2] = {1.0, 1.0};
	double fval = run_test<Beale, 2>(x);

	EXPECT_LT( std::fabs(x[0] - 3.0),  1e-3);
	EXPECT_LT( std::fabs(x[1] - 0.5), 1e-9);
	EXPECT_LT( std::fabs(fval), 1e-9);
}


struct JennrichSampson
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R fval = 0;
		for (int ii = 1; ii <= 10; ++ii) {
			double i = ii;
			R d = 2.0 + 2.0*i - (exp(i*x[0]) + exp(i*x[1]));
			fval += d*d;
		}
		return fval;
	}
};

TEST(Solver, JennrichSampson)
{
	double x[2] = {0.3, 0.4};
	double fval = run_test<JennrichSampson, 2>(x);

	EXPECT_LT( std::fabs(fval - 124.362), 0.001);
}

struct HelicalValley
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R theta = 1.0 / (2.0 * 3.141592653589793)
		          * atan(x[1] / x[0]);
		if (x[0] < 0) {
			theta += 0.5;
		}
		R d0 = 10.0 * (x[2] - 10.0 * theta);
		R d1 = 10.0 * (sqrt(x[0]*x[0] + x[1]*x[1]) - 1.0);
		R d2 = x[2];
		return d0*d0 + d1*d1 + d2*d2;
	}
};

TEST(Solver, HelicalValley)
{
	double x[3] = {-1.0, 0.0, 0.0};
	double fval = run_test<HelicalValley, 3>(x);

	EXPECT_LT( std::fabs(x[0] - 1.0),  1e-9);
	EXPECT_LT( std::fabs(x[1]), 1e-9);
	EXPECT_LT( std::fabs(x[2]), 1e-9);
	EXPECT_LT( std::fabs(fval), 1e-9);
}

struct Bard
{
	template<typename R>
	R operator()(const R* const x) const
	{
		double y[15] = {0.14, 0.18, 0.22, 0.25,
		                0.29, 0.32, 0.35, 0.39,
		                0.37, 0.58, 0.73, 0.96,
						1.34, 2.10, 4.39};
		R fval = 0;
		for (int ii = 1; ii <= 15; ++ii) {
			double u = ii;
			double v = 16.0 - ii;
			double w = std::min(u, v);
			R d = y[ii - 1] - (x[0] + u / (v * x[1] + w * x[2]));
			fval += d*d;
		}
		return fval;
	}
};

TEST(Solver, Bard)
{
	double x[3] = {1.0, 1.0, 1.0};
	double fval = run_test<Bard, 3>(x);

	EXPECT_LT( std::fabs(fval - 8.21487e-3), 1e-7);
}

struct Gaussian
{
	template<typename R>
	R operator()(const R* const x) const
	{
		double y[15] = {0.0009, 0.0044, 0.0175, 0.0540,
		                0.1295, 0.2420, 0.3521, 0.3989,
		                0.3521, 0.2420, 0.1295, 0.0540,
		                0.0175, 0.0044, 0.0009};
		R fval = 0;
		for (int ii = 1; ii <= 15; ++ii) {
			double t = (8.0 - ii) / 2.0;
			R tdiff = t - x[2];
			R d = x[0] * exp( -x[1]*tdiff*tdiff / 2.0) - y[ii - 1];
			fval += d*d;
		}
		return fval;
	}
};

TEST(Solver, Gaussian)
{
	double x[3] = {0.4, 1.0, 0.0};
	double fval = run_test<Gaussian, 3>(x);

	EXPECT_LT( std::fabs(fval - 1.12793e-8), 1e-12);
}

struct Meyer
{
	template<typename R>
	R operator()(const R* const x) const
	{
		double y[16] = {34780, 28610, 23650, 19630,
		                16370, 13720, 11540,  9744,
		                 8261,  7030,  6005,  5147,
		                 4427,  3820,  3307,  2872};
		R fval = 0;
		for (int ii = 1; ii <= 16; ++ii) {
			double t = 45.0 + 5.0 * ii;
			R d = x[0] * exp( x[1] / (t + x[2])) - y[ii - 1];
			fval += d*d;
		}
		return fval;
	}
};

TEST(Solver, Meyer)
{
	double x[3] = {0.02, 4000.0, 250.0};
	Solver solver;
	create_solver(&solver);
	if (solver.maximum_iterations < 500) {
		solver.maximum_iterations = 500;
	}
	double fval = run_test<Meyer, 3>(x, &solver);

	EXPECT_LT( std::fabs(fval - 87.9458), 1e-3);
}

template<int m = 10>
struct Gulf
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R fval = 0;
		for (int ii = 1; ii <= m; ++ii) {
			double t = ii / 100.0;
			double y = 25.0 + pow(-50.0 * log(t), 2.0 / 3.0);
			R d = exp( - pow( y * m * ii * x[1], x[2]) / x[0] ) - t;
			fval += d*d;
		}
		return fval;
	}
};

TEST(Solver, Gulf)
{
	double x[3] = {5, 2.5, 0.15};
	double fval = run_test<Gulf<3>, 3>(x);

	// The Gulf function does not evaluate to close to 0
	// at the globally optimal point. Hence these tests
	// are disabled.
	//EXPECT_LT( std::fabs(x[0] - 50.0), 1e-9);
	//EXPECT_LT( std::fabs(x[1] - 25.0), 1e-9);
	//EXPECT_LT( std::fabs(x[2] - 1.5),  1e-9);
	//EXPECT_LT( std::fabs(fval), 1e-9);
}

template<int m>
struct Box
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R fval = 0;
		for (int ii = 1; ii <= m; ++ii) {
			double t = 0.1 * ii;
			R d = exp(-t*x[0]) - exp(-t*x[1]) - x[2]*(exp(-t) - exp(-10.0*t));
			fval += d*d;
		}
		return fval;
	}
};

TEST(Solver, Box)
{
	double x[3] = {1.0, 10.0, 20.0};
	double fval = run_test<Box<10>, 3>(x);

	EXPECT_LT( std::fabs(fval), 1e-9);
}

struct PowellSingular
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R d0 = x[0] + 10 * x[1];
		R d1 = sqrt(5.0) * (x[2] - x[3]);
		R d2 = x[1] - 2.0*x[3];
		d2 = d2 * d2;
		R d3 = sqrt(10.0) * (x[0] - x[3]) * (x[0] - x[3]);
		return d0*d0 + d1*d1 + d2*d2 + d3*d3;
	}
};

TEST(Solver, PowellSingular)
{
	double x[4] = {3.0, -1.0, 0.0, 1.0};
	double fval = run_test<PowellSingular, 4>(x);

	EXPECT_LT( std::fabs(x[0]), 1e-3);  // Hard to end up with the variables
	EXPECT_LT( std::fabs(x[1]), 1e-3);  // very close to the origin.
	EXPECT_LT( std::fabs(x[2]), 1e-3);
	EXPECT_LT( std::fabs(x[3]), 1e-3);
	EXPECT_LT( std::fabs(fval), 1e-12);
}

struct Wood
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R f1 = 10.0 * (x[1] - x[0]*x[0]);
		R f2 = 1 - x[0];
		R f3 = sqrt(90.0) * (x[3] - x[2]*x[2]);
		R f4 = 1- x[2];
		R f5 = sqrt(10.0) * (x[1] + x[3] - 2.0);
		R f6 = 1.0 / sqrt(10.0) * (x[1] - x[3]);
		return f1*f1 + f2*f2 + f3*f3 + f4*f4;
	}
};

TEST(Solver, Wood)
{
	double x[4] = {-3.0, -1.0, -3.0, -1.0};
	double fval = run_test<Wood, 4>(x);

	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-8);
	EXPECT_LT( std::fabs(x[1] - 1.0), 1e-8);
	EXPECT_LT( std::fabs(x[2] - 1.0), 1e-8);
	EXPECT_LT( std::fabs(x[3] - 1.0), 1e-8);
	EXPECT_LT( std::fabs(fval), 1e-8);
}


//-----------------------------------------------------------------
// Test functions from TEST_OPT
// http://people.sc.fsu.edu/~jburkardt/m_src/test_opt/test_opt.html
//-----------------------------------------------------------------

// #29
struct GoldsteinPricePolynomial
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R a = x[0] + x[1] + 1.0;

		R b = 19.0 - 14.0 * x[0] + 3.0 * x[0] * x[0] - 14.0 * x[1]
			+ 6.0 * x[0] * x[1] + 3.0 * x[1] * x[1];

		R c = 2.0 * x[0] - 3.0 * x[1];

		R d = 18.0 - 32.0 * x[0] + 12.0 * x[0] * x[0] + 48.0 * x[1]
			- 36.0 * x[0] * x[1] + 27.0 * x[1] * x[1];

		return ( 1.0 + a * a * b ) * ( 30.0 + c * c * d );
	}
};

TEST(Solver, GoldsteinPricePolynomial)
{
	double x[2] = {-0.5, 0.25};
	// Only expect a local minimum where the gradient
	// is small.
	Solver solver;
	create_solver(&solver);
	solver.argument_improvement_tolerance = 0;
	solver.function_improvement_tolerance = 0;
	run_test<GoldsteinPricePolynomial, 2>(x, &solver);
}

// #30
struct BraninRCOS
{
	template<typename R>
	R operator()(const R* const x) const
	{
		const double pi = 3.141592653589793;
		const double a  = 1.0;
		const double d  = 6.0;
		const double e  = 10.0;
		const double b  = 5.1 / ( 4.0 * pi*pi );
		const double c  = 5.0 / pi;
		const double ff = 1.0 / ( 8.0 * pi );

		R expr = ( x[1] - b * x[0]*x[0] + c * x[0] - d );
		return a * expr * expr
			+ e * ( 1.0 - ff ) * cos ( x[0] ) + e;
	}
};

TEST(Solver, BraninRCOS)
{
	double x[2] = {-1.0, 1.0};
	// Only expect a local minimum where the gradient
	// is small.
	Solver solver;
	create_solver(&solver);
	solver.argument_improvement_tolerance = 0;
	solver.function_improvement_tolerance = 0;
	solver.gradient_tolerance = 1e-10;
	run_test<BraninRCOS, 2>(x, &solver);
}

// #34
struct SixHumpCamelBack
{
	template<typename R>
	R operator()(const R* const x) const
	{
		return ( 4.0 - 2.1 * x[0]*x[0] + x[0]*x[0]*x[0]*x[0] / 3.0 ) * x[0]*x[0]
	            + x[0] * x[1] + 4.0 * ( x[1]*x[1] - 1.0 ) * x[1]*x[1];
	}
};

TEST(Solver, SixHumpCamelBack)
{
	double x[2] = {-1.5, 0.5};
	// Only expect a local minimum where the gradient
	// is small.
	Solver solver;
	create_solver(&solver);
	solver.argument_improvement_tolerance = 0;
	solver.function_improvement_tolerance = 0;
	solver.gradient_tolerance = 1e-10;
	run_test<SixHumpCamelBack, 2>(x, &solver);
}


// #35
struct Shubert
{
	template<typename R>
	R operator()(const R* const x) const
	{
		R factor1 = 0.0;
		for (int i = 1; i<= 5; ++i) {
			double y = i;
			factor1 += y * cos( (y + 1.0) * x[0] + y );
		}

		R factor2 = 0.0;
		for (int i = 1; i<= 5; ++i) {
			double y = i;
			factor2 += y * cos( (y + 1.0) * x[1] + y );
		}

		return factor1 * factor2;
	}
};

TEST(Solver, Shubert)
{
	double x[2] = {0.5, 1.0};
	// Only expect a local minimum where the gradient
	// is small.
	Solver solver;
	create_solver(&solver);
	solver.argument_improvement_tolerance = 0;
	solver.function_improvement_tolerance = 0;
	solver.gradient_tolerance = 1e-10;
	run_test<Shubert, 2>(x, &solver);
}

// #37
struct Easom
{
	template<typename R>
	R operator()(const R* const x) const
	{
		const double pi = 3.141592653589793;
		R arg = - (x[0] - pi)*(x[0] - pi) - (x[1] - pi)*(x[1] - pi);
		return -cos(x[0]) * cos(x[1]) * exp(arg);
	}
};

TEST(Solver, Easom)
{
	Solver solver;
	create_solver(&solver);
	solver.maximum_iterations = 10000;

	double x[2] = {0.5, 1.0};
	run_test<Easom, 2>(x, &solver);

	// There seems to be other local minima, though.
	EXPECT_LT( std::fabs(x[0] - 3.141592653589793), 1e-8);
	EXPECT_LT( std::fabs(x[1] - 3.141592653589793), 1e-8);
}

// #38
struct Bohachevsky1
{
	// f = x(1) * x(1) - 0.3 * cos ( 3.0 * pi * x(1) ) ...
	//   + 2.0 * x(2) * x(2) - 0.4 * cos ( 4.0 * pi * x(2) ) ...
	//   + 0.7;
	template<typename R>
	R operator()(const R* const x) const
	{
		return x[0] * x[0] - 0.3 * cos(3.0 * 3.141592 * x[0])
			+ 2.0 * x[1] * x[1] - 0.4 * cos(4.0 * 3.141592 * x[1])
			+ 0.7;
	}
};

// #39
struct Bohachevsky2
{
	template<typename R>
	R operator()(const R* const x) const
	{
		return x[0]*x[0] + 2.0*x[1]*x[1]
			-0.3*cos(3.0 * 3.141592 * x[0]) *
				 cos( 4.0 * 3.141592 * x[1] ) + 0.3;
	}
};

// #40
struct Bohachevsky3
{
	template<typename R>
	R operator()(const R* const x) const
	{
		return x[0]*x[0] + 2.0*x[1]*x[1]
			- 0.3 * cos ( 3.0 * 3.141592 * x[0] )
			+ cos ( 4.0 * 3.141592 * x[1] ) + 0.3;
	}
};

TEST(Solver, Bohachevsky)
{
	double x[2];

	// Only expect a local minimum where the gradient
	// is small.
	Solver solver;
	create_solver(&solver);
	solver.argument_improvement_tolerance = 0;
	solver.function_improvement_tolerance = 0;


	if (solver.gradient_tolerance >= 1e-16) {
		solver.gradient_tolerance = 1e-8;
	}

	x[0] = 0.5;
	x[1] = 1.0;
	run_test<Bohachevsky1, 2>(x, &solver);

	x[0] = 0.6;
	x[1] = 1.3;
	run_test<Bohachevsky2, 2>(x, &solver);

	x[0] = 0.5;
	x[1] = 1.0;
	run_test<Bohachevsky3, 2>(x, &solver);
}


// #41
struct Colville
{
	template<typename R>
	R operator()(const R* const x) const
	{
		return 100.0 * (x[1] - x[0]*x[0]) * (x[1] - x[0]*x[0])
		+ (1.0 - x[0]) * (1.0 - x[0])
		+ 90.0 * (x[3] - x[2]*x[2]) * (x[3] - x[2]*x[2])
		+ (1.0 - x[2]) * (1.0 - x[2])
		+ 10.1 * ( (x[1] - 1.0 ) * (x[1] - 1.0)
			+ (x[3] - 1.0) * (x[3] - 1.0) )
		+ 19.8 * (x[1] - 1.0) * (x[3] - 1.0);
	}
};

TEST(Solver, Colville)
{
	double x[4] = {-0.5, 1.0, -0.5, -1.0};
	double fval = run_test<Colville, 4>(x);

	EXPECT_LT( std::fabs(x[0] - 1.0), 1e-8);
	EXPECT_LT( std::fabs(x[1] - 1.0), 1e-8);
	EXPECT_LT( std::fabs(x[2] - 1.0), 1e-8);
	EXPECT_LT( std::fabs(x[3] - 1.0), 1e-8);
}

// #42
struct Powell3D
{
	template<typename R>
	R operator()(R* x) const
	{
		R term = 0.0;
		if (to_double(x[1]) != 0.0) {
			R arg = (x[0] + 2.0*x[1] + x[2]) / x[1];
			term = exp(-arg*arg);
		}

		return 3.0
			- 1.0 / (1.0 + (x[0] - x[1])*(x[0] - x[1]))
			- sin( 0.5 * 3.141592653589793 * x[1] * x[2])
			- term;
	}
};

TEST(Solver, Powell3D)
{
	// There appears to be numerical problems at
	// x = (-1, -1, 3).
	//
	// Symbolic computation shows that the gradient indeed is 0
	// at (-1, -1, 3), but the function returns a gradient whose
	// maximum element is over 1e-8.
	//

	double x[3] = {-1.0, -1.0, 3.0};
	Function f;
	f.add_variable(x, 3);
	f.add_term(new AutoDiffTerm<Powell3D, 3>(new Powell3D()), x);
	Eigen::VectorXd xvec(3);

	xvec[0] = -1.0;
	xvec[1] = -1.0;
	xvec[2] =  3.0;
	Eigen::VectorXd g;
	f.evaluate(xvec, &g);
	std::cerr << "g(-1, -1, 3)  = (" << g.transpose() << ")" << std::endl;

	x[0] = 0.0;
	x[1] = 1.0;
	x[2] = 2.0;
	double fval = run_test<Powell3D, 3>(x);

	xvec[0] = x[0];
	xvec[1] = x[1];
	xvec[2] = x[2];
	f.evaluate(xvec, &g);
	std::printf("x = (%.16e, %.16e, %.16e\n", x[0], x[1], x[2]);
	std::cerr << "g = (" << g.transpose() << ")" << std::endl;

	// The webpage states that the optimal point is
	// (1, 1, 1), but that seems incorrect.
	//EXPECT_LT( std::fabs(x[0] - 1.0), 1e-8);
	//EXPECT_LT( std::fabs(x[1] - 1.0), 1e-8);
	//EXPECT_LT( std::fabs(x[2] - 1.0), 1e-8);
}

// #43
struct Himmelblau
{
	template<typename R>
	R operator()(const R* const x) const
	{
		// f = ( x(1)^2 + x(2) - 11.0 )^2 + ( x(1) + x(2)^2 - 7.0 )^2;
		R d1 = x[0]*x[0] + x[1]      - 11.0;
		R d2 = x[0]      + x[1]*x[1] - 7.0;
		return d1*d1 + d2*d2;
	}
};

TEST(Solver, Himmelblau)
{
	double x[2] = {-1.3, 2.7};
	double fval = run_test<Himmelblau, 2>(x);

	EXPECT_LT( std::fabs(fval), 1e-8);
}
