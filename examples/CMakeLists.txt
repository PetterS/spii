# Author: petter.strandmark@gmail.com (Petter Strandmark)

# Also depend on the header files so that they appear in IDEs.
FILE(GLOB SPII_HEADERS *.h)

MACRO (SPII_EXAMPLE NAME)
  ADD_EXECUTABLE(${NAME} ${NAME}.cpp)
  TARGET_LINK_LIBRARIES(${NAME} spii)
ENDMACRO (SPII_EXAMPLE)

SPII_EXAMPLE(fit_cir)
SPII_EXAMPLE(fit_gaussian)
SPII_EXAMPLE(linear_programming)
