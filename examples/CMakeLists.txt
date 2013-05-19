# Author: petter.strandmark@gmail.com (Petter Strandmark)

MACRO (SPII_EXAMPLE NAME)
	ADD_EXECUTABLE(${NAME}
	               ${NAME}.cpp
	               ${SPII_HEADERS})
	TARGET_LINK_LIBRARIES(${NAME} spii)
ENDMACRO (SPII_EXAMPLE)

#SPII_EXAMPLE(bundle_adjustment)
SPII_EXAMPLE(ellipse)
SPII_EXAMPLE(fit_cir)
SPII_EXAMPLE(fit_gaussian)
SPII_EXAMPLE(lennard_jones)
SPII_EXAMPLE(linear_programming)
