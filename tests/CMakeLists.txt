# Author: petter.strandmark@gmail.com (Petter Strandmark)

# Also depend on the header files so that they appear in IDEs.
FILE(GLOB SPII_HEADERS *.h)



IF (${BUILD_TESTING})
  MACRO (SPII_TEST NAME)
    ADD_EXECUTABLE(test_${NAME} test_${NAME}.cpp)
    TARGET_LINK_LIBRARIES(test_${NAME} spii)
    TARGET_LINK_LIBRARIES(test_${NAME} meschach)

    ADD_TEST(NAME test_${NAME}
             COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${NAME})
  ENDMACRO (SPII_TEST)

  SPII_TEST(function)
  SPII_TEST(global)
  SPII_TEST(interval)
  SPII_TEST(meschach)
  SPII_TEST(solver)
  SPII_TEST(suite_large)
  SPII_TEST(suite_lbfgs)
  SPII_TEST(suite_newton)
  SPII_TEST(suite_nelder_mead)
  SPII_TEST(suite_pattern_search)
  SPII_TEST(term)

  # The NIST tests are not part of the test suite that has to succeed, because
  # the solver may end up in a local minimum.
  ADD_EXECUTABLE(test_nist test_nist.cpp)
    TARGET_LINK_LIBRARIES(test_nist spii)
    TARGET_LINK_LIBRARIES(test_nist meschach)

ENDIF (${BUILD_TESTING})
