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
	include/imageio/AnnotatedImage.hpp
	include/imageio/AnnotatedImageSource.hpp
	include/imageio/Annotation.hpp
	include/imageio/Annotations.hpp
	include/imageio/AnnotationSink.hpp
	include/imageio/AnnotationSource.hpp
	include/imageio/BobotAnnotationSink.hpp
	include/imageio/BobotAnnotationSource.hpp
	include/imageio/CameraImageSource.hpp
	include/imageio/DirectoryImageSink.hpp
	include/imageio/DirectoryImageSource.hpp
	include/imageio/DlibImageSource.hpp
	include/imageio/EmptyAnnotationSource.hpp
	include/imageio/ImageSink.hpp
	include/imageio/ImageSource.hpp
	include/imageio/OrderedAnnotatedImageSource.hpp
	include/imageio/SingleAnnotationSink.hpp
	include/imageio/SingleAnnotationSource.hpp
	include/imageio/VideoImageSink.hpp
	include/imageio/VideoImageSource.hpp
)
set(SOURCE
	src/imageio/BobotAnnotationSink.cpp
	src/imageio/BobotAnnotationSource.cpp
	src/imageio/CameraImageSource.cpp
	src/imageio/DirectoryImageSink.cpp
	src/imageio/DirectoryImageSource.cpp
	src/imageio/DlibImageSource.cpp
	src/imageio/OrderedAnnotatedImageSource.cpp
	src/imageio/SingleAnnotationSink.cpp
	src/imageio/SingleAnnotationSource.cpp
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

