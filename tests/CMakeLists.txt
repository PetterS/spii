# Author: petter.strandmark@gmail.com (Petter Strandmark)

ADD_LIBRARY(Catch main_test_program.cpp)

MACRO (SPII_TEST NAME)
	ADD_EXECUTABLE(${NAME}${EXECUTABLE_EXTENSION}
	               ${NAME}.cpp
	               ${SPII_HEADERS})

	TARGET_LINK_LIBRARIES(${NAME}${EXECUTABLE_EXTENSION} spii Catch)
	SET_PROPERTY(TARGET ${NAME}${EXECUTABLE_EXTENSION}         PROPERTY FOLDER "Tests")
	ADD_TEST(NAME ${NAME}${EXECUTABLE_EXTENSION}
	         COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
ENDMACRO (SPII_TEST)

FILE(GLOB TEST_FILES test_*.cpp)
FILE(GLOB MESCHACH_TEST test_meschach*.cpp)
LIST(REMOVE_ITEM TEST_FILES ${MESCHACH_TEST})
FILE(GLOB STRING_TEST test_string*.cpp)
LIST(REMOVE_ITEM TEST_FILES ${STRING_TEST})
LIST(REMOVE_ITEM TEST_FILES test_nist_global.cpp)

FOREACH (TEST_FILE ${TEST_FILES})
	GET_FILENAME_COMPONENT(TEST_NAME ${TEST_FILE} NAME_WE)
	MESSAGE("-- Adding test: " ${TEST_NAME})
	SPII_TEST(${TEST_NAME})
ENDFOREACH()

# Meschach test does not link to spii.
ADD_EXECUTABLE(test_meschach${EXECUTABLE_EXTENSION} test_meschach.cpp)
	TARGET_LINK_LIBRARIES(test_meschach${EXECUTABLE_EXTENSION} meschach Catch)
	ADD_TEST(NAME meschach${EXECUTABLE_EXTENSION}
	       COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_meschach${EXECUTABLE_EXTENSION})
	SET_PROPERTY(TARGET test_meschach${EXECUTABLE_EXTENSION}         PROPERTY FOLDER "Tests")

# String test does not link to spii.
ADD_EXECUTABLE(test_string${EXECUTABLE_EXTENSION} test_string.cpp)
	TARGET_LINK_LIBRARIES(test_string${EXECUTABLE_EXTENSION} Catch)
	ADD_TEST(NAME string${EXECUTABLE_EXTENSION}
	       COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_string${EXECUTABLE_EXTENSION})
	SET_PROPERTY(TARGET test_string${EXECUTABLE_EXTENSION}         PROPERTY FOLDER "Tests")

#
# Valgrind (memcheck) testing.
#
find_program(VALGRIND NAMES valgrind PATH /usr/bin /usr/local/bin)
if (VALGRIND)
	message("-- Valgrind found; configuring memcheck tests.")
	macro(VALGRIND_TEST NAME)
		add_test(NAME memcheck_${NAME}
		         COMMAND ${VALGRIND} --error-exitcode=1 --leak-check=full ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${NAME})
	endmacro(VALGRIND_TEST)

	# Run shorter tests though memcheck.
	VALGRIND_TEST(function)
	VALGRIND_TEST(function_serializer)
	VALGRIND_TEST(solver)
	VALGRIND_TEST(spii)
	VALGRIND_TEST(term)
	VALGRIND_TEST(term_factory)
endif (VALGRIND)

