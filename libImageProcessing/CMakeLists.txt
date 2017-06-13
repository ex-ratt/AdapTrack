SET(SUBPROJECT_NAME ImageProcessing)
PROJECT(${SUBPROJECT_NAME})

MESSAGE(STATUS "Configuring ${SUBPROJECT_NAME}")

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core imgproc)

INCLUDE_DIRECTORIES("include")
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

ADD_LIBRARY(${SUBPROJECT_NAME}
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
TARGET_LINK_LIBRARIES(${SUBPROJECT_NAME}
	${OpenCV_LIBS}
)

INSTALL(TARGETS ${SUBPROJECT_NAME}
	LIBRARY DESTINATION lib
)
INSTALL(DIRECTORY include/
	DESTINATION include
)
