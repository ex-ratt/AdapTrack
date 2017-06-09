set(SUBPROJECT_NAME ImageIO)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
find_package(Boost 1.48.0 COMPONENTS system filesystem REQUIRED)
find_package(OpenCV 2.4.3 REQUIRED core highgui)

# source and header files
set(HEADERS
	include/imageio/BobotLandmarkSink.hpp
	include/imageio/BobotLandmarkSource.hpp
	include/imageio/CameraImageSource.hpp
	include/imageio/DirectoryImageSink.hpp
	include/imageio/DirectoryImageSource.hpp
	include/imageio/DlibImageSource.hpp
	include/imageio/EmptyLandmarkSource.hpp
	include/imageio/ImageSink.hpp
	include/imageio/ImageSource.hpp
	include/imageio/LabeledImageSource.hpp
	include/imageio/Landmark.hpp
	include/imageio/LandmarkCollection.hpp
	include/imageio/LandmarkSink.hpp
	include/imageio/LandmarkSource.hpp
	include/imageio/OrderedLabeledImageSource.hpp
	include/imageio/RectLandmark.hpp
	include/imageio/RepeatingFileImageSource.hpp
	include/imageio/SingleLandmarkSink.hpp
	include/imageio/SingleLandmarkSource.hpp
	include/imageio/VideoImageSink.hpp
	include/imageio/VideoImageSource.hpp
)
set(SOURCE
	src/imageio/BobotLandmarkSink.cpp
	src/imageio/BobotLandmarkSource.cpp
	src/imageio/CameraImageSource.cpp
	src/imageio/DirectoryImageSink.cpp
	src/imageio/DirectoryImageSource.cpp
	src/imageio/DlibImageSource.cpp
	src/imageio/LandmarkCollection.cpp
	src/imageio/OrderedLabeledImageSource.cpp
	src/imageio/RectLandmark.cpp
	src/imageio/RepeatingFileImageSource.cpp
	src/imageio/SingleLandmarkSink.cpp
	src/imageio/SingleLandmarkSource.cpp
	src/imageio/VideoImageSink.cpp
	src/imageio/VideoImageSource.cpp
)

include_directories("include")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS ${SUBPROJECT_NAME}
	LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

