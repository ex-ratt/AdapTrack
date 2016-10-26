SET( SUBPROJECT_NAME benchmarkApp ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(Boost 1.48.0 COMPONENTS system filesystem program_options REQUIRED)

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core highgui video)

# source and header files
SET(HEADERS
	AlgorithmData.hpp
	Benchmark.hpp
)
SET(SOURCE
	AlgorithmData.cpp
	Benchmark.cpp
	BenchmarkRunner.cpp
)

# add dependencies
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${ImageProcessing_SOURCE_DIR}/include)
include_directories(${Classification_SOURCE_DIR}/include)
include_directories(${SVM_SOURCE_DIR}/include)
include_directories(${Detection_SOURCE_DIR}/include)

# make executable
add_executable(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Detection SVM Classification ImageProcessing ImageIO Logging ${OpenCV_LIBS} ${Boost_LIBRARIES})
