# Author: petter.strandmark@gmail.com (Petter Strandmark)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

IF (COMMAND cmake_policy)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF (COMMAND cmake_policy)

PROJECT(SPII C CXX)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


# To get a more static build, try the following line on Mac and Linux:
# SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})

# Default locations to search for on various platforms.
LIST(APPEND SEARCH_LIBS /usr/lib)
LIST(APPEND SEARCH_LIBS /usr/local/lib)
LIST(APPEND SEARCH_LIBS /usr/local/homebrew/lib) # Mac OS X
LIST(APPEND SEARCH_LIBS /opt/local/lib)

LIST(APPEND SEARCH_HEADERS /usr/include)
LIST(APPEND SEARCH_HEADERS /usr/local/include)
LIST(APPEND SEARCH_HEADERS /usr/local/homebrew/include) # Mac OS X
LIST(APPEND SEARCH_HEADERS /opt/local/include)

# Locations to search for Eigen
SET(EIGEN_SEARCH_HEADERS ${SEARCH_HEADERS})
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/include/eigen3) # Ubuntu 10.04's default location.
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/local/include/eigen3)
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/local/homebrew/include/eigen3)  # Mac OS X
LIST(APPEND EIGEN_SEARCH_HEADERS /opt/local/var/macports/software/eigen3/opt/local/include/eigen3) # Mac OS X

# Eigen
MESSAGE("-- Check for Eigen 3.0")
FIND_PATH(EIGEN_INCLUDE NAMES Eigen/Core PATHS ${EIGEN_SEARCH_HEADERS})
IF (NOT EXISTS ${EIGEN_INCLUDE})
  MESSAGE(FATAL_ERROR "Can't find Eigen. Try passing -DEIGEN_INCLUDE=...")
ENDIF (NOT EXISTS ${EIGEN_INCLUDE})
MESSAGE("-- Found Eigen 3.0: ${EIGEN_INCLUDE}")

ENABLE_TESTING()

OPTION(BUILD_TESTING
       "Enable tests"
       ON)

IF (${BUILD_TESTING})

  # Find Google's testing library headers
  FIND_PATH(GTEST_INCLUDE NAMES gtest/gtest.h PATHS ${SEARCH_HEADERS})
  IF (NOT EXISTS ${GTEST_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find Google's testing headers. Please specify: "
            "-DGTEST_INCLUDE=...")
  ENDIF (NOT EXISTS ${GTEST_INCLUDE})
  MESSAGE("-- Found Google Test headers in: ${GTEST_INCLUDE}")

  # Find Google's testing library
  FIND_LIBRARY(GTEST_LIB NAMES gtest PATHS ${SEARCH_LIBS})
  IF (NOT EXISTS ${GTEST_LIB})
    MESSAGE(FATAL_ERROR
              "Can't find Google Test gtest. Please specify: "
              "-DGTEST_LIB=...")
  ENDIF (NOT EXISTS ${GTEST_LIB})
  MESSAGE("-- Found Google Test: ${GTEST_LIB}")

  # Find Google's testing main library
  FIND_LIBRARY(GTEST_MAIN_LIB NAMES gtest_main PATHS ${SEARCH_LIBS})
  IF (NOT EXISTS ${GTEST_MAIN_LIB})
    MESSAGE(FATAL_ERROR
              "Can't find Google Test Main gtest_main. Please specify: "
              "-DGTEST_MAIN_LIB=...")
  ENDIF (NOT EXISTS ${GTEST_MAIN_LIB})
  MESSAGE("-- Found Google Test Main: ${GTEST_MAIN_LIB}")
ENDIF (${BUILD_TESTING})

# Multithreading using OpenMP
OPTION(OPENMP
       "Enable multi-threading (requires OpenMP)"
       ON)

IF (${OPENMP})
  FIND_PACKAGE(OpenMP)
  IF(${OPENMP_FOUND})
    MESSAGE("-- Found OpenMP.")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ADD_DEFINITIONS(-DUSE_OPENMP)
  ELSE(${OPENMP_FOUND})
    MESSAGE("-- Can't find OpenMP. Continuing without it.")
  ENDIF(${OPENMP_FOUND})
ENDIF (${OPENMP})


INCLUDE_DIRECTORIES(
  include
  thirdparty/FADBAD++
  ${EIGEN_INCLUDE}
  )

FILE(GLOB SPII_HEADERS ${CMAKE_SOURCE_DIR}/include/spii/*.h)
INSTALL(FILES ${SPII_HEADERS} DESTINATION include/spii)

# Library dependencies
SET (SPII_LIBRARY_DEPENDENCIES)

# OpenMP needs to be included as a library on some platforms.
IF (${OPENMP_FOUND})
  IF (NOT MSVC)
    LIST(APPEND SPII_LIBRARY_DEPENDENCIES gomp)
  ENDIF (NOT MSVC)
ENDIF (${OPENMP_FOUND})

# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.
IF (NOT CMAKE_BUILD_TYPE)
  MESSAGE("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
ELSE (NOT CMAKE_BUILD_TYPE)
  IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE("\n=================================================================================")
    MESSAGE("\n-- Build type: Debug. Performance will be terrible!")
    MESSAGE("-- Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
    MESSAGE("\n=================================================================================")
  ENDIF (CMAKE_BUILD_TYPE STREQUAL "Debug")
ENDIF (NOT CMAKE_BUILD_TYPE)

# Set the default flags to an empty string.
SET (SPII_CXX_FLAGS)

IF (CMAKE_BUILD_TYPE STREQUAL "Release")
  IF (CMAKE_COMPILER_IS_GNUCXX)
      # Linux
      IF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        SET (SPII_CXX_FLAGS "${SPII_CXX_FLAGS} -march=native -mtune=native")
      ENDIF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      # Mac OS X
      IF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        SET (SPII_CXX_FLAGS "${SPII_CXX_FLAGS} -fast -msse3")
      ENDIF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  ENDIF (CMAKE_COMPILER_IS_GNUCXX)
  SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${SPII_CXX_FLAGS}"
       CACHE STRING "Release mode flags to the C++ Compiler" FORCE)
ENDIF (CMAKE_BUILD_TYPE STREQUAL "Release")

# GCC is not strict enough by default, so enable most of the warnings.
IF ("${UNIX}")
  SET(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused -Wno-unused-parameter")
ENDIF ("${UNIX}")

OPTION(BUILD_EXAMPLES "Build examples" ON)

IF (${BUILD_EXAMPLES})
  MESSAGE("-- Build the examples.")
  ADD_SUBDIRECTORY(examples)
ELSE (${BUILD_EXAMPLES})
  MESSAGE("-- Do not build any example.")
ENDIF (${BUILD_EXAMPLES})

ADD_SUBDIRECTORY(source)
ADD_SUBDIRECTORY(tests)

