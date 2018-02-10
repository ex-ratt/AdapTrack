SET(SUBPROJECT_NAME Tracking)
PROJECT(${SUBPROJECT_NAME})

MESSAGE(STATUS "Configuring ${SUBPROJECT_NAME}")

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core imgproc)

INCLUDE_DIRECTORIES("include")
INCLUDE_DIRECTORIES(${Detection_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${SVM_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${Classification_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${ImageProcessing_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

ADD_LIBRARY(${SUBPROJECT_NAME}
	src/tracking/MultiTracker.cpp
	src/tracking/SingleTracker.cpp
	src/tracking/filtering/ParticleFilter.cpp
	src/tracking/filtering/TargetState.cpp
)
TARGET_LINK_LIBRARIES(${SUBPROJECT_NAME}
	Detection
	SVM
	Classification
	ImageProcessing
	${OpenCV_LIBS}
)

INSTALL(TARGETS ${SUBPROJECT_NAME}
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
INSTALL(DIRECTORY include/
	DESTINATION include
)
