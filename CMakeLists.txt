project(AdapTrack)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Libraries
add_subdirectory(libLogging)            # text logging to terminal or files
add_subdirectory(libImageIO)            # reading and writing of images, videos, and annotations
add_subdirectory(libImageProcessing)    # image pyramids, filters, feature extraction
add_subdirectory(libClassification)     # binary and probabilistic classification
add_subdirectory(libSvm)                # wrapper around libSVM
add_subdirectory(libDetection)          # detection
add_subdirectory(libTracking)           # tracking

# Applications
add_subdirectory(DetectorTrainer)       # trains detectors based on aggregated channel features and support vector machines
add_subdirectory(Tracker)               # tracks objects using detector and particle filter
add_subdirectory(VideoPlayer)           # plays videos, shows bounding boxes, and stores resulting videos with bounding boxes

# Old libraries (that will be removed/replaced)
add_subdirectory(libImageLogging)       # image logging
add_subdirectory(libCondensation)       # particle filter based tracking

# Old applications (that will be removed/replaced)
add_subdirectory(benchmarkApp)          # Benchmark app for feature extractors and classifiers in a tracking-like online learning scenario.
add_subdirectory(trackingBenchmarkApp)  # Benchmark app for adaptive condensation tracking.
add_subdirectory(adaptiveTrackingApp)   # Adaptive tracking app.
add_subdirectory(headTrackingApp)       # Adaptive head tracking app.
