/*
 * ResizingFilter.hpp
 *
 *  Created on: 15.03.2013
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_RESIZINGFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_RESIZINGFILTER_HPP_

#include "imageprocessing/filtering/ImageFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace imageprocessing {
namespace filtering {

/**
 * Filter that resizes images to a certain size.
 */
class ResizingFilter : public ImageFilter {
public:

	/**
	 * Constructs a new resizing filter.
	 *
	 * @param[in] size The size of the filtered images.
	 * @param[in] interpolation The interpolation method (see last parameter of cv::resize).
	 */
	explicit ResizingFilter(cv::Size size, int interpolation = cv::INTER_AREA);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	cv::Size size;     ///< The size of the filtered images.
	int interpolation; ///< The interpolation method.
};

} /* namespace filtering */
} /* namespace imageprocessing */
#endif /* IMAGEPROCESSING_FILTERING_RESIZINGFILTER_HPP_ */
