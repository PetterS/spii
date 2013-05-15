
#ifdef USE_OPENMP
	#include <omp.h>
#endif

#include <spii/spii.h>

namespace spii
{
double wall_time()
{
	#ifdef USE_OPENMP
		return omp_get_wtime();
	#else
		return 0.0;
	#endif
}
}
