/*
 * AnnotatedImage.hpp
 *
 *  Created on: Jun 9, 2017
 *      Author: ex-ratt
 */

#ifndef IMAGEIO_ANNOTATEDIMAGE_HPP_
#define IMAGEIO_ANNOTATEDIMAGE_HPP_

#include "imageio/Annotations.hpp"
#include "opencv2/core/core.hpp"

namespace imageio {

/**
 * Image with annotations.
 */
struct AnnotatedImage {
	cv::Mat image; ///< Image.
	Annotations annotations; ///< Rectangular annotations.
};

} /* namespace imageio */

#endif /* IMAGEIO_ANNOTATEDIMAGE_HPP_ */
