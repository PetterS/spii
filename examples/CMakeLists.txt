# Author: petter.strandmark@gmail.com (Petter Strandmark)

# Also depend on the header files so that they appear in IDEs.
FILE(GLOB SPII_HEADERS *.h)

MACRO (SPII_EXAMPLE NAME)
  ADD_EXECUTABLE(${NAME} ${NAME}.cpp)
  TARGET_LINK_LIBRARIES(${NAME} spii)
ENDMACRO (SPII_EXAMPLE)

#SPII_EXAMPLE(bundle_adjustment)
SPII_EXAMPLE(ellipse)
SPII_EXAMPLE(fit_cir)
SPII_EXAMPLE(fit_gaussian)
SPII_EXAMPLE(lennard_jones)
SPII_EXAMPLE(linear_programming)
