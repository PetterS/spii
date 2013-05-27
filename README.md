This is a library for unconstrained minimization of smooth functions with a large number of variables. I wrote this to get a better grasp of nonlinear optimization. I used the book ny Nocedal and Wright [1] as a reference.

[![Build Status](https://travis-ci.org/PetterS/spii.png)](https://travis-ci.org/PetterS/spii)

Features
--------
* Newton's method 
    * Bunch-Kaufman-Parlett factorization and block-diagonal modification. This is a robust method of dealing with nonconvex functions.
    * The sparse solver uses repeated diagonal modification of the hessian for nonconvex problems. This simple method seems to work well, but can require several Cholesky factorizations per iteration and is not as robust as B-K-P.
* Sparse Cholesky factorization using Eigen (included) if the problem is large and sparse.
* L-BFGS.
* Nelder-Mead for nondifferentiable problems.
* Automatic differentiation to compute gradient and hessian using FADBAD++ (included).
* Multi-threaded using OpenMP.
* Change of variables (experimental).
* Global optimization with interval arithmetic (experimental).

Compilation
-----------
Everything needed to compile the library, examples and tests using CMake should be included.
All tests pass with the following compilers:
* Visual Studio 2012
* GCC 4.6 (Ubuntu)
* GCC 4.7 (Ubuntu)
* Clang 3.2 (Ubuntu)
Earlier compilers will probably not work.

You can check travis.yml for the commands used to build the library and run all tests on Ubuntu.
It is even easier on Windows. The status of the automatic builds using gcc and Clang is [![Build Status](https://travis-ci.org/PetterS/spii.png)](https://travis-ci.org/PetterS/spii).


Benchmarks
----------
The solver comes with extensive benchmarks.

* The NIST collection of non-linear least-squares problems. http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
* Test functions from More et al. [2].
* TEST_OPT http://people.sc.fsu.edu/~jburkardt/m_src/test_opt/test_opt.html

The Newton solver and the L-BFGS solver pass all of these tests. The NIST collection was very challenging and required block-diagonal robust factorization and handling of numerically hard problem instances. Note that non-linear least-squares problems have a special structure and are best solved with custom code, for example Ceres Solver.

References
----------
1. Nocedal and Wright, *Numerical Optimization*, Springer, 2006.
2. Jorge J. More, Burton S. Garbow and Kenneth E. Hillstrom, *Testing unconstrained optimization software*, Transactions on Mathematical Software 7(1):17-41, 1981.
