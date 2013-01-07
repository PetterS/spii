This is a library for unconstrained minimization of smooth functions with a large number of variables. I wrote this to get a better grasp of nonlinear optimization. I used the book ny Nocedal and Wright [1] as a reference.

Features
--------
* Newton's method with repeated diagonal modification of the hessian for nonconvex problems. This simple method seems to work well, but can require several Cholesky factorizations per iteration.
* Sparse Cholesky factorization using Eigen (included) if the problem is large and sparse.
* L-BFGS.
* Nelder-Mead for nondifferentiable problems.
* Automatic differentiation to compute gradient and hessian using FADBAD++ (included).
* Multi-threaded using OpenMP.
* Change of variables (experimental).

Compilation
-----------
Everything needed to compile the library and examples using CMake should be included. The unit tests require Google's testing framework.
All tests pass with the following compilers:
* Visual Studio 2010
* Visual Studio 2012
* GCC 4.5 (Cygwin)
* GCC 4.7 (Ubuntu)
* Clang 3.2 (Ubuntu)
* Clang 3.1 (Cygwin)
Earlier compilers will probably not work.

Benchmarks
----------
The tests include the first 14 problems from a standard set of difficult small problems [2].
The Rosenbrock function is minimized in 573µs using Newton's method and in 659µs using L-BFGS.

Note that the examples include linear programming and least-squares problems. Of course, a specialized solver should be used when encountering these problems (see e.g. Ceres Solver for nonlinear least squares).

References
----------
1. Nocedal and Wright, *Numerical Optimization*, Springer, 2006.
2. Jorge J. More, Burton S. Garbow and Kenneth E. Hillstrom, *Testing unconstrained optimization software*, Transactions on Mathematical Software 7(1):17-41, 1981.
