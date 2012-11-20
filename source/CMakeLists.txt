# Author: petter.strandmark@gmail.com (Petter Strandmark)

SET(SPII_INTERNAL_SRC
    term.cpp
)

# Also depend on the header files so that they appear in IDEs.
FILE(GLOB SPII_HEADERS *.h)

SET(SPII_LIBRARY_SOURCE
    ${SPII_INTERNAL_SRC}
    ${SPII_HEADERS})

ADD_LIBRARY(spii STATIC ${SPII_LIBRARY_SOURCE})
TARGET_LINK_LIBRARIES(spii ${SPII_LIBRARY_DEPENDENCIES})

INSTALL(TARGETS spii
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)


