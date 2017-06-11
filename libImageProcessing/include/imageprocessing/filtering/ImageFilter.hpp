/*
 * ImageFilter.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_IMAGEFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_IMAGEFILTER_HPP_

#include "opencv2/core/core.hpp"

namespace imageprocessing {
namespace filtering {

/**
 * Filter of images.
 */
class ImageFilter {
public:

	/**
	 * Constructs a new image filter.
	 */
	ImageFilter() {}

	virtual ~ImageFilter() {}

	/**
	 * Applies this filter to an image, writing the filtered image data into a newly created image.
	 *
	 * @param[in] image The image that should be filtered.
	 * @return The filtered image.
	 */
	cv::Mat applyTo(const cv::Mat& image) const {
		cv::Mat filtered;
		applyTo(image, filtered);
		return filtered;
	}

	/**
	 * Applies this filter to an image, writing the filtered image data into a new image.
	 *
	 * @param[in] image The image that should be filtered.
	 * @param[out] filtered The image for writing the filtered data into.
	 * @return The filtered image.
	 */
	virtual cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const = 0;

	/**
	 * Applies this filter to an image, writing the filtered data into the image itself.
	 *
	 * @param[in,out] image The image that should be filtered.
	 */
	virtual void applyInPlace(cv::Mat& image) const {
		image = applyTo(image);
	}
};

} /* namespace filtering */
} /* namespace imageprocessing */
#endif /* IMAGEPROCESSING_FILTERING_IMAGEFILTER_HPP_ */
