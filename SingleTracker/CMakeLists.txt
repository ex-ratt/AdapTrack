SET(SUBPROJECT_NAME SingleTracker)
PROJECT(${SUBPROJECT_NAME})

MESSAGE(STATUS "Configuring ${SUBPROJECT_NAME}")

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core highgui)

INCLUDE_DIRECTORIES(${Tracking_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${SVM_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${Classification_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${ImageProcessing_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${ImageIO_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

ADD_EXECUTABLE(${SUBPROJECT_NAME}
  SingleTracker.cpp
)
TARGET_LINK_LIBRARIES(${SUBPROJECT_NAME}
	Tracking
	SVM
	Classification
	ImageProcessing
	ImageIO
	${OpenCV_LIBS}
)

INSTALL(TARGETS ${SUBPROJECT_NAME}
	RUNTIME DESTINATION bin
)
