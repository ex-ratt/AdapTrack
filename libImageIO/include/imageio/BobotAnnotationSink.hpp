/*
 * BobotAnnotationSink.hpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#ifndef BOBOTANNOTATIONSINK_HPP_
#define BOBOTANNOTATIONSINK_HPP_

#include "imageio/AnnotationSink.hpp"
#include "imageio/ImageSource.hpp"
#include <fstream>
#include <memory>

namespace imageio {

class Landmark;
class ImageSource;

/**
 * Annotation sink that writes annotations to a file using the Bonn Benchmark on Tracking (BoBoT) format.
 */
class BobotAnnotationSink : public AnnotationSink {
public:

	/**
	 * Constructs a new BoBoT annotation sink with explicit image dimensions.
	 *
	 * @param[in] videoFilename The name of the video file the annotations belong to (will be written in the first line of the file).
	 * @param[in] imageWidth The width of the images.
	 * @param[in] imageHeight The height of the images.
	 */
	BobotAnnotationSink(const std::string& videoFilename, int imageWidth, int imageHeight);

	/**
	 * Constructs a new BoBoT annotation sink that takes the image dimensions from an image source.
	 *
	 * @param[in] videoFilename The name of the video file the annotations belong to (will be written in the first line of the file).
	 * @param[in] imageSource The source of the images. Is assumed to be at the position the added annotation belong to.
	 */
	BobotAnnotationSink(const std::string& videoFilename, std::shared_ptr<ImageSource> imageSource);

	~BobotAnnotationSink();

	BobotAnnotationSink(BobotAnnotationSink& other) = delete;

	BobotAnnotationSink operator=(BobotAnnotationSink rhs) = delete;

	bool isOpen();

	void open(const std::string& filename);

	void close();

	void add(Annotations annotations);

private:

	const std::string videoFilename; ///< The name of the video file the annotations belong to (will be written in the first line of the file).
	double imageWidth; ///< The width of the images.
	double imageHeight; ///< The height of the images.
	std::shared_ptr<ImageSource> imageSource; ///< The source of the images. Is assumed to be at the position the added annotation belong to.
	std::ofstream output; ///< The file output stream.
	unsigned int index; ///< The index of the next annotation that is written to the file.
};

} /* namespace imageio */
#endif /* BOBOTANNOTATIONSINK_HPP_ */
