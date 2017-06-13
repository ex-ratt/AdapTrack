SET( SUBPROJECT_NAME Classification ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core)

# source and header files
SET(HEADERS
	include/classification/AgeBasedExampleManagement.hpp
	include/classification/BinaryClassifier.hpp
	include/classification/ClassifierTrainer.hpp
	include/classification/ConfidenceBasedExampleManagement.hpp
	include/classification/ExampleManagement.hpp
	include/classification/HistogramIntersectionKernel.hpp
	include/classification/IncrementalClassifierTrainer.hpp
	include/classification/IncrementalLinearSvmTrainer.hpp
	include/classification/Kernel.hpp
	include/classification/KernelVisitor.hpp
	include/classification/LinearKernel.hpp
	include/classification/PolynomialKernel.hpp
	include/classification/ProbabilisticClassifier.hpp
	include/classification/ProbabilisticSupportVectorMachine.hpp
	include/classification/PseudoIncrementalClassifierTrainer.hpp
	include/classification/PseudoProbabilisticSvmTrainer.hpp
	include/classification/RbfKernel.hpp
	include/classification/SupportVectorMachine.hpp
	include/classification/UnlimitedExampleManagement.hpp
)
SET(SOURCE
	src/classification/AgeBasedExampleManagement.cpp
	src/classification/ConfidenceBasedExampleManagement.cpp
	src/classification/ProbabilisticSupportVectorMachine.cpp
	src/classification/SupportVectorMachine.cpp
)

include_directories("include")

# add dependencies
include_directories(${OpenCV_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} ${OpenCV_LIBS})

install(TARGETS ${SUBPROJECT_NAME}
	LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
