SET(SUBPROJECT_NAME Detection)
PROJECT(${SUBPROJECT_NAME})

MESSAGE(STATUS "Configuring ${SUBPROJECT_NAME}")

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core)

INCLUDE_DIRECTORIES("include")
INCLUDE_DIRECTORIES(${Classification_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${ImageProcessing_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${ImageIO_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

ADD_LIBRARY(${SUBPROJECT_NAME}
	src/detection/AggregatedFeaturesDetector.cpp
	src/detection/DetectorTester.cpp
	src/detection/DetectorTrainer.cpp
	src/detection/NonMaximumSuppression.cpp
)
TARGET_LINK_LIBRARIES(${SUBPROJECT_NAME}
	Classification
	ImageProcessing
	ImageIO
	${OpenCV_LIBS}
)

INSTALL(TARGETS ${SUBPROJECT_NAME}
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
install(DIRECTORY include/
	DESTINATION include
)
