# Author: petter.strandmark@gmail.com (Petter Strandmark)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)
PROJECT(SPII C CXX)

OPTION(BUILD_TESTING "Enable tests" OFF)
OPTION(BUILD_EXAMPLES "Build examples" ON)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Default locations to search for on various platforms.
LIST(APPEND SEARCH_LIBS /usr/lib)
LIST(APPEND SEARCH_LIBS /usr/local/lib)
LIST(APPEND SEARCH_LIBS /usr/local/homebrew/lib) # Mac OS X
LIST(APPEND SEARCH_LIBS /opt/local/lib)

LIST(APPEND SEARCH_HEADERS /usr/include)
LIST(APPEND SEARCH_HEADERS /usr/local/include)
LIST(APPEND SEARCH_HEADERS /usr/local/homebrew/include) # Mac OS X
LIST(APPEND SEARCH_HEADERS /opt/local/include)

ENABLE_TESTING()
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
  thirdparty/Eigen
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
ENDIF (NOT CMAKE_BUILD_TYPE)

#
# gcc settings.
#
IF (CMAKE_COMPILER_IS_GNUCXX)
  MESSAGE("-- Enabling C++11 support for gcc.")
  IF (${BUILD_TESTING})
    ADD_DEFINITIONS("-std=gnu++0x")
  ELSE (${BUILD_TESTING})
    ADD_DEFINITIONS("-std=c++0x")
  ENDIF (${BUILD_TESTING})

  # GCC is not strict enough by default, so enable most of the warnings.
  ADD_DEFINITIONS("-Wall -Wextra -Wno-sign-compare -Wno-unused-parameter")
ENDIF (CMAKE_COMPILER_IS_GNUCXX)

#
# Clang settings
#
IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  MESSAGE("-- Enabling C++11 support for clang.")
  ADD_DEFINITIONS("-std=c++11")

  # Enable warnings for Clang.
  ADD_DEFINITIONS("-Wall -Wextra -Wno-sign-compare -Wno-unused-parameter")
ENDIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

#
# MSVC settings
#
IF (MSVC)
  # Disable deprecation warning for standard functions.
  ADD_DEFINITIONS("/wd4996")
ENDIF (MSVC)


IF (CMAKE_BUILD_TYPE STREQUAL "Release")
  IF (CMAKE_COMPILER_IS_GNUCXX)
      MESSAGE("-- Optimizing for release mode.")
      # Linux
      IF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        MESSAGE("-- Linux")
        ADD_DEFINITIONS("-march=native -mtune=native")
      ENDIF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      # Cygwin
      IF (${CMAKE_SYSTEM_NAME} STREQUAL "CYGWIN")
        MESSAGE("-- Cygwin")
        ADD_DEFINITIONS("-march=native -mtune=native")
      ENDIF (${CMAKE_SYSTEM_NAME} STREQUAL "CYGWIN")
      # Mac OS X
      IF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        MESSAGE("-- MacOS")
        ADD_DEFINITIONS("-fast -msse3")
      ENDIF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  ENDIF (CMAKE_COMPILER_IS_GNUCXX)
ENDIF (CMAKE_BUILD_TYPE STREQUAL "Release")

IF (${BUILD_EXAMPLES})
  MESSAGE("-- Build the examples.")
  ADD_SUBDIRECTORY(examples)
ELSE (${BUILD_EXAMPLES})
  MESSAGE("-- Do not build any example.")
ENDIF (${BUILD_EXAMPLES})

ADD_SUBDIRECTORY(data)
ADD_SUBDIRECTORY(source)
ADD_SUBDIRECTORY(tests)

