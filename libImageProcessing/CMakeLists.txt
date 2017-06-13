SET( SUBPROJECT_NAME ImageProcessing ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core imgproc)

# source and header files
SET(HEADERS
	include/imageprocessing/ImagePyramid.hpp
	include/imageprocessing/ImagePyramidLayer.hpp
	include/imageprocessing/Patch.hpp
	include/imageprocessing/Version.hpp
	include/imageprocessing/VersionedImage.hpp
	include/imageprocessing/extraction/AggregatedFeaturesExtractor.hpp
	include/imageprocessing/extraction/ExactFhogExtractor.hpp
	include/imageprocessing/extraction/FeatureExtractor.hpp
	include/imageprocessing/filtering/AggregationFilter.hpp
	include/imageprocessing/filtering/BgrToLuvConverter.hpp
	include/imageprocessing/filtering/BoxConvolutionFilter.hpp
	include/imageprocessing/filtering/ChainedFilter.hpp
	include/imageprocessing/filtering/ChannelExtractor.hpp
	include/imageprocessing/filtering/ColorSpaceConversionFilter.hpp
	include/imageprocessing/filtering/ConversionFilter.hpp
	include/imageprocessing/filtering/ConvolutionFilter.hpp
	include/imageprocessing/filtering/FhogAggregationFilter.hpp
	include/imageprocessing/filtering/FhogFilter.hpp
	include/imageprocessing/filtering/FilterTree.hpp
	include/imageprocessing/filtering/FpdwFeaturesFilter.hpp
	include/imageprocessing/filtering/GradientFilter.hpp
	include/imageprocessing/filtering/GradientHistogramFilter.hpp
	include/imageprocessing/filtering/GradientMagnitudeFilter.hpp
	include/imageprocessing/filtering/GradientOrientationFilter.hpp
	include/imageprocessing/filtering/GrayscaleFilter.hpp
	include/imageprocessing/filtering/HistogramFilter.hpp
	include/imageprocessing/filtering/ImageFilter.hpp
	include/imageprocessing/filtering/ResizingFilter.hpp
	include/imageprocessing/filtering/TriangularConvolutionFilter.hpp
)
SET(SOURCE
	src/imageprocessing/ImagePyramid.cpp
	src/imageprocessing/Version.cpp
	src/imageprocessing/extraction/AggregatedFeaturesExtractor.cpp
	src/imageprocessing/extraction/ExactFhogExtractor.cpp
	src/imageprocessing/filtering/AggregationFilter.cpp
	src/imageprocessing/filtering/BgrToLuvConverter.cpp
	src/imageprocessing/filtering/BoxConvolutionFilter.cpp
	src/imageprocessing/filtering/ChainedFilter.cpp
	src/imageprocessing/filtering/ChannelExtractor.cpp
	src/imageprocessing/filtering/ColorSpaceConversionFilter.cpp
	src/imageprocessing/filtering/ConversionFilter.cpp
	src/imageprocessing/filtering/ConvolutionFilter.cpp
	src/imageprocessing/filtering/FhogAggregationFilter.cpp
	src/imageprocessing/filtering/FhogFilter.cpp
	src/imageprocessing/filtering/FilterTree.cpp
	src/imageprocessing/filtering/FpdwFeaturesFilter.cpp
	src/imageprocessing/filtering/GradientFilter.cpp
	src/imageprocessing/filtering/GradientHistogramFilter.cpp
	src/imageprocessing/filtering/GradientMagnitudeFilter.cpp
	src/imageprocessing/filtering/GradientOrientationFilter.cpp
	src/imageprocessing/filtering/GrayscaleFilter.cpp
	src/imageprocessing/filtering/HistogramFilter.cpp
	src/imageprocessing/filtering/ResizingFilter.cpp
	src/imageprocessing/filtering/TriangularConvolutionFilter.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${OpenCV_INCLUDE_DIRS} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
target_link_libraries(${SUBPROJECT_NAME} ${OpenCV_LIBS})

install(TARGETS ${SUBPROJECT_NAME}
	LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
