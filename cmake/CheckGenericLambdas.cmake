INCLUDE(CheckCXXSourceCompiles)

CHECK_CXX_SOURCE_COMPILES("int main() { auto lambda = [](auto x, auto y) { return x + y; };  }" USE_GENERIC_LAMBDAS)

IF (USE_GENERIC_LAMBDAS)
	MESSAGE("-- Generic lambdas (C++14) are supported by the compiler.")
ELSE ()
	MESSAGE("-- Generic lambdas (C++14) are not supported by the compiler.")
ENDIF()
