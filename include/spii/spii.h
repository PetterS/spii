#ifndef SPII_H
#define SPII_H

#ifdef USE_OPENMP
	#include <omp.h>
#endif

namespace {
double wall_time()
{
	#ifdef USE_OPENMP
		return omp_get_wtime();
	#else
		return 0.0;
	#endif
}
}

#ifdef _WIN32
#	ifdef spii_EXPORTS
#		define SPII_API __declspec(dllexport)
#		define SPII_API_EXTERN_TEMPLATE
#	else
#		define SPII_API __declspec(dllimport)
#		define SPII_API_EXTERN_TEMPLATE extern
#	endif
#else
#	define SPII_API
#	define SPII_API_EXTERN_TEMPLATE
#endif // WIN32

#endif
