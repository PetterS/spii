# Author: petter.strandmark@gmail.com (Petter Strandmark)

MACRO (SPII_TEST NAME)
	ADD_EXECUTABLE(test_${NAME} 
	               test_${NAME}.cpp
	               ${SPII_HEADERS})

	TARGET_LINK_LIBRARIES(test_${NAME} spii)

	ADD_TEST(NAME ${NAME}
	         COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${NAME})
ENDMACRO (SPII_TEST)

SPII_TEST(function)
SPII_TEST(function_serializer)
SPII_TEST(global)
SPII_TEST(interval)
SPII_TEST(nist)
SPII_TEST(solver)
SPII_TEST(spii)
SPII_TEST(suite_large)
SPII_TEST(suite_lbfgs)
SPII_TEST(suite_newton)
SPII_TEST(suite_nelder_mead)
SPII_TEST(suite_pattern_search)
SPII_TEST(term)
SPII_TEST(term_factory)

# Meschach test does not link to spii.
ADD_EXECUTABLE(test_meschach test_meschach.cpp)
	TARGET_LINK_LIBRARIES(test_meschach meschach)
	ADD_TEST(NAME meschach
	       COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_meschach)

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
