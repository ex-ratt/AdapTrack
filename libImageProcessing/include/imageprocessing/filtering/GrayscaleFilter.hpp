/*
 * GrayscaleFilter.hpp
 *
 *  Created on: 19.02.2013
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_GRAYSCALEFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_GRAYSCALEFILTER_HPP_

#include "imageprocessing/filtering/ImageFilter.hpp"

namespace imageprocessing {
namespace filtering {

/**
 * Image filter that converts images to grayscale. If the input image does have only one channel, then the data
 * will be copied to the output image without any conversion. Otherwise, the image data is assumed to be BGR.
 */
class GrayscaleFilter : public ImageFilter {
public:

	/**
	 * Constructs a new grayscale filter.
	 */
	GrayscaleFilter();

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	void applyInPlace(cv::Mat& image) const;
};

} /* namespace filtering */
} /* namespace imageprocessing */
#endif /* IMAGEPROCESSING_FILTERING_GRAYSCALEFILTER_HPP_ */
