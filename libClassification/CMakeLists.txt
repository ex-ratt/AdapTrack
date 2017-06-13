SET(SUBPROJECT_NAME Classification)
PROJECT(${SUBPROJECT_NAME})

MESSAGE(STATUS "Configuring ${SUBPROJECT_NAME}")

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core)

INCLUDE_DIRECTORIES("include")
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

ADD_LIBRARY(${SUBPROJECT_NAME}
	src/classification/AgeBasedExampleManagement.cpp
	src/classification/ConfidenceBasedExampleManagement.cpp
	src/classification/ProbabilisticSupportVectorMachine.cpp
	src/classification/SupportVectorMachine.cpp
)
TARGET_LINK_LIBRARIES(${SUBPROJECT_NAME}
	${OpenCV_LIBS}
)

INSTALL(TARGETS ${SUBPROJECT_NAME}
	LIBRARY DESTINATION lib
)
INSTALL(DIRECTORY include/
	DESTINATION include
)
