SET(SUBPROJECT_NAME ImageIO)
PROJECT(${SUBPROJECT_NAME})

MESSAGE(STATUS "Configuring ${SUBPROJECT_NAME}")

FIND_PACKAGE(Boost 1.48.0 REQUIRED system filesystem)
FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core highgui)

INCLUDE_DIRECTORIES("include")
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

ADD_LIBRARY(${SUBPROJECT_NAME}
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
TARGET_LINK_LIBRARIES(${SUBPROJECT_NAME}
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
)

INSTALL(TARGETS ${SUBPROJECT_NAME}
	LIBRARY DESTINATION lib
)
INSTALL(DIRECTORY include/
	DESTINATION include
)
