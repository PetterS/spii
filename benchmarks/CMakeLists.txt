# Author: petter.strandmark@gmail.com (Petter Strandmark)

MACRO (SPII_BENCHMARK NAME)
	ADD_EXECUTABLE(benchmark_${NAME} 
	               benchmark_${NAME}.cpp
	               ${SPII_HEADERS})

	TARGET_LINK_LIBRARIES(benchmark_${NAME} spii)
ENDMACRO (SPII_BENCHMARK)

SPII_BENCHMARK(bundle)
SPII_BENCHMARK(function)
SPII_BENCHMARK(solver)
