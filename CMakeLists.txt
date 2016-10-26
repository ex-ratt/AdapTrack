set(PROJECT_NAME AdapTrack)
project(${PROJECT_NAME})
cmake_minimum_required(VERSION 2.8)

# g++ needs a compiler flag to use the (experimental) C++11 support
# TODO: * We could test for VS/gcc/unsupported compilers and warn
#       * This check doesn't interfere with the compilation on windows, but is not necessary there.
#       * We should stop and throw an error if we're on gcc and no c++11 support is found
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++11 HAS_CXX11_FLAG)
if (HAS_CXX11_FLAG)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# Libraries
add_subdirectory(libLogging)            # text logging to terminal or files
add_subdirectory(libImageLogging)       # image logging
add_subdirectory(libImageIO)            # reading and writing of images, videos, and annotations
add_subdirectory(libImageProcessing)    # image pyramids, filters, feature extraction
add_subdirectory(libClassification)     # binary and probabilistic classification
add_subdirectory(libSvm)                # wrapper around libSVM
add_subdirectory(libDetection)          # detection
add_subdirectory(libCondensation)       # particle filter based tracking

# Applications
add_subdirectory(DetectorTrainer)       # trains detectors based on aggregated channel features and support vector machines
add_subdirectory(VideoPlayer)           # plays videos, shows bounding boxes, and stores resulting videos with bounding boxes

# Old applications (that will be replaced)
add_subdirectory(benchmarkApp)          # Benchmark app for feature extractors and classifiers in a tracking-like online learning scenario.
add_subdirectory(trackingBenchmarkApp)  # Benchmark app for adaptive condensation tracking.
add_subdirectory(adaptiveTrackingApp)   # Adaptive tracking app.
add_subdirectory(headTrackingApp)       # Adaptive head tracking app.

