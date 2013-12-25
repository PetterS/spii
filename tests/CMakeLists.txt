# Author: petter.strandmark@gmail.com (Petter Strandmark)

MACRO (SPII_TEST NAME)
	ADD_EXECUTABLE(${NAME} 
	               ${NAME}.cpp
	               ${SPII_HEADERS})

	TARGET_LINK_LIBRARIES(${NAME} spii)

	ADD_TEST(NAME ${NAME}
	         COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
ENDMACRO (SPII_TEST)

FILE(GLOB TEST_FILES test_*.cpp)
FILE(GLOB MESCHACH_TEST test_meschach*.cpp)
LIST(REMOVE_ITEM TEST_FILES ${MESCHACH_TEST})
FILE(GLOB STRING_TEST test_string*.cpp)
LIST(REMOVE_ITEM TEST_FILES ${STRING_TEST})
FOREACH (TEST_FILE ${TEST_FILES})
	GET_FILENAME_COMPONENT(TEST_NAME ${TEST_FILE} NAME_WE)
	MESSAGE("-- Adding test: " ${TEST_NAME})
	SPII_TEST(${TEST_NAME})
ENDFOREACH()

# Meschach test does not link to spii.
ADD_EXECUTABLE(test_meschach test_meschach.cpp)
	TARGET_LINK_LIBRARIES(test_meschach meschach)
	ADD_TEST(NAME meschach
	       COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_meschach)

# String test does not link to spii.
ADD_EXECUTABLE(test_string test_string.cpp)
	ADD_TEST(NAME string
	       COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_string)

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

#
# Cppcheck testing.
#
find_program(CPPCHECK NAMES cppcheck PATH /usr/bin /usr/local/bin)
if (CPPCHECK)
	message("-- Cppcheck found; configuring test.")
	add_test(NAME cppcheck
	         COMMAND ${CPPCHECK} ${CMAKE_SOURCE_DIR} -i ${CMAKE_SOURCE_DIR}/thirdparty/ -q --error-exitcode=1)
endif (CPPCHECK)
