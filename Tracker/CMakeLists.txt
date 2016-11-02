SET( SUBPROJECT_NAME Tracker )
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(Boost 1.48.0 COMPONENTS system filesystem program_options REQUIRED)

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core highgui video)

# add dependencies
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${ImageProcessing_SOURCE_DIR}/include)
include_directories(${Classification_SOURCE_DIR}/include)
include_directories(${SVM_SOURCE_DIR}/include)
include_directories(${Detection_SOURCE_DIR}/include)
include_directories(${Condensation_SOURCE_DIR}/include)
include_directories(${Tracking_SOURCE_DIR}/include)

# make executable
add_executable(${SUBPROJECT_NAME}
  Tracker.cpp
)
target_link_libraries(${SUBPROJECT_NAME}
	Tracking
	Condensation
	Detection
	SVM
	Classification
	ImageProcessing
	ImageIO
	Logging
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
)
