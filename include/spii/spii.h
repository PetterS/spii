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

#endif
