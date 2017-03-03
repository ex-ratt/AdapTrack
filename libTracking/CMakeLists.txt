set(SUBPROJECT_NAME Tracking)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set( ${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set( ${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
find_package(Boost 1.48.0 REQUIRED)
find_package(OpenCV 2.4.3 REQUIRED core imgproc video)

include_directories("include")

# add dependencies
include_directories(${Detection_SOURCE_DIR}/include)
include_directories(${SVM_SOURCE_DIR}/include)
include_directories(${Classification_SOURCE_DIR}/include)
include_directories(${ImageProcessing_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME}
	src/tracking/Tracker.cpp
	src/tracking/filtering/ParticleFilter.cpp
	src/tracking/filtering/TargetState.cpp
)
target_link_libraries(${SUBPROJECT_NAME}
	Detection
	SVM
	Classification
	ImageProcessing
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
)

install(TARGETS ${SUBPROJECT_NAME}
	LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

