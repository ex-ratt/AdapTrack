/*
 * ColorSpaceConversionFilter.hpp
 *
 *  Created on: 08.08.2013
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_COLORSPACECONVERSIONFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_COLORSPACECONVERSIONFILTER_HPP_

#include "imageprocessing/filtering/ImageFilter.hpp"

namespace imageprocessing {
namespace filtering {

/**
 * Image filter that converts images from one color space to another.
 */
class ColorSpaceConversionFilter : public ImageFilter {
public:

	/**
	 * Constructs a new color space conversion filter.
	 *
	 * @param[in] conversion The conversion code, see cv::cvtColor for details.
	 */
	explicit ColorSpaceConversionFilter(int conversion);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	int conversion; ///< The conversion code, see cv::cvtColor for details.
};

} /* namespace filtering */
} /* namespace imageprocessing */
#endif /* IMAGEPROCESSING_FILTERING_COLORSPACECONVERSIONFILTER_HPP_ */
