project(AdapTrack)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS ON)

# Libraries
add_subdirectory(libLogging)         # text logging to terminal or files
add_subdirectory(libImageIO)         # reading and writing of images, videos, and annotations
add_subdirectory(libImageProcessing) # image pyramids, filters, feature extraction
add_subdirectory(libClassification)  # binary and probabilistic classification
add_subdirectory(libSvm)             # wrapper around libSVM
add_subdirectory(libDetection)       # detection
add_subdirectory(libTracking)        # tracking

# Applications
add_subdirectory(DetectorTrainer)    # trains detectors based on aggregated channel features and support vector machines
add_subdirectory(SingleTracker)      # tracks a single object without prior knowledge
add_subdirectory(MultiTracker)       # tracks multiple objects using detector and particle filter
