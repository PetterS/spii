# Author: petter.strandmark@gmail.com (Petter Strandmark)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.4)
PROJECT(SPII C CXX)

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
SET(CTEST_OUTPUT_ON_FAILURE TRUE)

# Multithreading using OpenMP
OPTION(OPENMP
       "Enable multi-threading (requires OpenMP)"
       ON)

# For Clang-based tools.
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
  thirdparty/boost
  SYSTEM thirdparty/Catch
  thirdparty/Eigen
  SYSTEM thirdparty/meschach
  SYSTEM thirdparty/ceres-solver-data
  )

FILE(GLOB SPII_HEADERS ${CMAKE_SOURCE_DIR}/include/spii/*.h)
FILE(GLOB SPII_THIRD_PARTY_HEADERS ${CMAKE_SOURCE_DIR}/include/spii-thirdparty/*.h)
INSTALL(FILES ${SPII_HEADERS} DESTINATION include/spii)
INSTALL(FILES ${SPII_THIRD_PARTY_HEADERS} DESTINATION include/spii-thirdparty)

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
IF (NOT CMAKE_BUILD_TYPE)
  MESSAGE("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
ENDIF (NOT CMAKE_BUILD_TYPE)

# C++11 support.
include(EnableCPP11.cmake)

#
# Clang and GCC settings
#
IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR CMAKE_COMPILER_IS_GNUCXX)
  # No warnings for C. The Meschach library contains really old code.
  # -fPIC for building a shared library.
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -w -fpic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-compare -Wno-unused-parameter -fpic")
ENDIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR CMAKE_COMPILER_IS_GNUCXX)

#
# MSVC settings
#
IF (MSVC)
  # Disable deprecation warning for standard functions.
  ADD_DEFINITIONS("/wd4996")
  # To get rid of annoying min and max macros if windows.h
  # is included.
  ADD_DEFINITIONS("-DNOMINMAX=1")
  # Macro redefinitions.
  ADD_DEFINITIONS("/wd4005")
ENDIF (MSVC)

#
# Packaging
#
SET(CPACK_GENERATOR ZIP)
SET(CPACK_PACKAGE_FILE_NAME "Spii")
INCLUDE(CPack)

ADD_SUBDIRECTORY(thirdparty/meschach)
ADD_SUBDIRECTORY(data)
ADD_SUBDIRECTORY(benchmarks)
ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(large_tests)
ADD_SUBDIRECTORY(source)
ADD_SUBDIRECTORY(tests)
