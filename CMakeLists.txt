PROJECT(AdapTrack)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(BUILD_SHARED_LIBS ON)

# Libraries
ADD_SUBDIRECTORY(libImageIO)         # reading and writing of images, videos, and annotations
ADD_SUBDIRECTORY(libImageProcessing) # image pyramids, filters, feature extraction
ADD_SUBDIRECTORY(libClassification)  # binary and probabilistic classification
ADD_SUBDIRECTORY(libSvm)             # wrapper around libSVM
ADD_SUBDIRECTORY(libDetection)       # detection
ADD_SUBDIRECTORY(libTracking)        # tracking

# Applications
ADD_SUBDIRECTORY(DetectorTrainer)    # trains detectors based on aggregated channel features and support vector machines
ADD_SUBDIRECTORY(SingleTracker)      # tracks a single object without prior knowledge
ADD_SUBDIRECTORY(MultiTracker)       # tracks multiple objects using detector and particle filter
