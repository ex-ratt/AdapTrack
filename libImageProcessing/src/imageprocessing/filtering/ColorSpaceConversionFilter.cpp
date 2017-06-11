/*
 * ColorSpaceConversionFilter.cpp
 *
 *  Created on: 08.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/filtering/ColorSpaceConversionFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using cv::Mat;

namespace imageprocessing {
namespace filtering {

ColorSpaceConversionFilter::ColorSpaceConversionFilter(int conversion) : conversion(conversion) {}

Mat ColorSpaceConversionFilter::applyTo(const Mat& image, Mat& filtered) const {
	cv::cvtColor(image, filtered, conversion);
	return filtered;
}

} /* namespace filtering */
} /* namespace imageprocessing */
