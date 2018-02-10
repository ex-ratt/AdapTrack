PROJECT(AdapTrack)
CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(BUILD_SHARED_LIBS OFF)

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
