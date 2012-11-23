# Author: petter.strandmark@gmail.com (Petter Strandmark)

# Also depend on the header files so that they appear in IDEs.
FILE(GLOB SPII_HEADERS *.h)



IF (${BUILD_TESTING})
  MACRO (SPII_TEST NAME)
    ADD_EXECUTABLE(test_${NAME} test_${NAME}.cpp)
    TARGET_LINK_LIBRARIES(test_${NAME} spii ${GTEST_LIB} ${GTEST_MAIN_LIB})
    INCLUDE_DIRECTORIES(${GTEST_INCLUDE})
    ADD_TEST(NAME test_${NAME}
             COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${NAME}
             ${CMAKE_SOURCE_DIR}/data)
  ENDMACRO (SPII_TEST)

  SPII_TEST(function)
  SPII_TEST(solver)
  SPII_TEST(suite)
  SPII_TEST(suite_large)
  SPII_TEST(term)

ENDIF (${BUILD_TESTING})
