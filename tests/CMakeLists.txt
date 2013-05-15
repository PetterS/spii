# Author: petter.strandmark@gmail.com (Petter Strandmark)

# Also depend on the header files so that they appear in IDEs.
FILE(GLOB SPII_HEADERS *.h)



IF (${BUILD_TESTING})
  MACRO (SPII_TEST NAME)
    ADD_EXECUTABLE(test_${NAME} test_${NAME}.cpp)
    TARGET_LINK_LIBRARIES(test_${NAME} spii)

    ADD_TEST(NAME ${NAME}
             COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${NAME})
  ENDMACRO (SPII_TEST)

  SPII_TEST(function)
  SPII_TEST(global)
  SPII_TEST(interval)
  SPII_TEST(nist)
  SPII_TEST(solver)
  SPII_TEST(suite_large)
  SPII_TEST(suite_lbfgs)
  SPII_TEST(suite_newton)
  SPII_TEST(suite_nelder_mead)
  SPII_TEST(suite_pattern_search)
  SPII_TEST(term)

  # Meschach test does not link to spii.
  ADD_EXECUTABLE(test_meschach test_meschach.cpp)
    TARGET_LINK_LIBRARIES(test_meschach meschach)
    ADD_TEST(NAME meschach
             COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_meschach)

ENDIF (${BUILD_TESTING})
