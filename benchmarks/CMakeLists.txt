# Author: petter.strandmark@gmail.com (Petter Strandmark)

MACRO (SPII_BENCHMARK NAME)
	ADD_EXECUTABLE(${NAME} 
	               ${NAME}.cpp
	               ${SPII_HEADERS})

	TARGET_LINK_LIBRARIES(${NAME} spii)
	SET_PROPERTY(TARGET ${NAME}         PROPERTY FOLDER "Benchmarks")
ENDMACRO (SPII_BENCHMARK)

FILE(GLOB BENCHMARK_FILES benchmark_*.cpp)
FOREACH (BENCHMARK_FILE ${BENCHMARK_FILES})
	GET_FILENAME_COMPONENT(BENCHMARK_NAME ${BENCHMARK_FILE} NAME_WE)
	MESSAGE("-- Adding benchmark: " ${BENCHMARK_NAME})
	SPII_BENCHMARK(${BENCHMARK_NAME})
ENDFOREACH()
