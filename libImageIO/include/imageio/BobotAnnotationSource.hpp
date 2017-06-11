/*
 * BobotAnnotationSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef BOBOTANNOTATIONSOURCE_HPP_
#define BOBOTANNOTATIONSOURCE_HPP_

#include "imageio/AnnotationSource.hpp"
#include "imageio/ImageSource.hpp"
#include "opencv2/core/core.hpp"
#include <string>
#include <vector>
#include <memory>

namespace imageio {

/**
 * Annotation source that reads from a Bonn Benchmark on Tracking (BoBoT) file. Uses the associated image
 * source to determine the size of the images.
 */
class BobotAnnotationSource : public AnnotationSource {
public:

	/**
	 * Constructs a new BoBoT annotation source with explicit image dimensions.
	 *
	 * @param[in] filename The name of the file containing the annotations in BoBoT format.
	 * @param[in] imageWidth The width of the images.
	 * @param[in] imageHeight The height of the images.
	 */
	BobotAnnotationSource(const std::string& filename, int imageWidth, int imageHeight);

	/**
	 * Constructs a new BoBoT annotation source that takes the image dimensions from an image source.
	 *
	 * @param[in] filename The name of the file containing the annotations in BoBoT format.
	 * @param(in] imageSource The source of the images. Is assumed to be at the same position as this source.
	 */
	BobotAnnotationSource(const std::string& filename, std::shared_ptr<ImageSource> imageSource);

	/**
	 * @return The name/path of the video file associated with this source (first line of file).
	 */
	const std::string& getVideoFilename() const;

	void reset();

	bool next();

	Annotations getAnnotations() const;

private:

	/**
	 * Reads the target positions for each frame from the given file.
	 *
	 * @param[in] filename The name of the file containing the annotation data in BoBoT format.
	 */
	void readPositions(const std::string& filename);

	mutable int imageWidth; ///< The width of the images.
	mutable int imageHeight; ///< The height of the images.
	std::shared_ptr<ImageSource> imageSource; ///< The source of the images. Is assumed to be at the same position as this source.
	std::string videoFilename; ///< The name/path of the video file associated with this source (first line of file).
	std::vector<cv::Rect_<float>> positions; ///< The target positions inside each image.
	int index; ///< The index of the current target position.
};

} /* namespace imageio */
#endif /* BOBOTANNOTATIONSOURCE_HPP_ */
