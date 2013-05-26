# Author: petter.strandmark@gmail.com (Petter Strandmark)

SET(SPII_INTERNAL_SRC
	function.cpp
	function_serializer.cpp
	solver.cpp
	solver_global.cpp
	solver_line_search.cpp
	solver_lbfgs.cpp
	solver_nelder_mead.cpp
	solver_newton.cpp
	solver_newton_factorization.cpp
	solver_pattern_search.cpp
	solver_stopping_criteria.cpp
	spii.cpp
	term.cpp
	term_factory.cpp
)

SET(SPII_LIBRARY_SOURCE
	${SPII_INTERNAL_SRC}
	${SPII_HEADERS})

ADD_LIBRARY(spii SHARED ${SPII_LIBRARY_SOURCE})
TARGET_LINK_LIBRARIES(spii ${SPII_LIBRARY_DEPENDENCIES})
TARGET_LINK_LIBRARIES(spii meschach)

INSTALL(TARGETS spii
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
