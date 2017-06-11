/*
 * Annotation.hpp
 *
 *  Created on: Jun 9, 2017
 *      Author: ex-ratt
 */

#ifndef IMAGEIO_ANNOTATION_HPP_
#define IMAGEIO_ANNOTATION_HPP_

#include "opencv2/core/core.hpp"

namespace imageio {

/**
 * Rectangular annotation.
 */
struct Annotation {

	/**
	 * Constructs a new annotation.
	 *
	 * @param[in] bounds Bounding box.
	 * @param[in] fuzzy Flag that indicates whether this annotation is neither positive, nor negative.
	 */
	explicit Annotation(cv::Rect bounds, bool fuzzy = false) : bounds(bounds), fuzzy(fuzzy) {}

	/**
	 * Constructs a new annotation.
	 *
	 * @param[in] bounds Bounding box.
	 * @param[in] fuzzy Flag that indicates whether this annotation is neither positive, nor negative.
	 */
	explicit Annotation(cv::Rect_<float> bounds, bool fuzzy = false) : bounds(), fuzzy(fuzzy) {
		this->bounds.x = static_cast<int>(std::round(bounds.tl().x));
		this->bounds.y = static_cast<int>(std::round(bounds.tl().y));
		this->bounds.width = static_cast<int>(std::round(bounds.br().x)) - this->bounds.x;
		this->bounds.height = static_cast<int>(std::round(bounds.br().y)) - this->bounds.y;
	}

	/**
	 * Constructs a new annotation.
	 *
	 * @param[in] bounds Bounding box.
	 * @param[in] fuzzy Flag that indicates whether this annotation is neither positive, nor negative.
	 */
	explicit Annotation(cv::Rect_<double> bounds, bool fuzzy = false) : bounds(), fuzzy(fuzzy) {
		this->bounds.x = static_cast<int>(std::round(bounds.tl().x));
		this->bounds.y = static_cast<int>(std::round(bounds.tl().y));
		this->bounds.width = static_cast<int>(std::round(bounds.br().x)) - this->bounds.x;
		this->bounds.height = static_cast<int>(std::round(bounds.br().y)) - this->bounds.y;
	}

	/**
	 * Adjusts the sizes of the annotation according to the given aspect ratio. If the aspect ratio of
	 * the bounding box diverges from the given one, then either the width or height is increased.
	 *
	 * @param[in] aspectRatio Aspect ratio of the window (width / height).
	 */
	void adjustSize(double aspectRatio) {
		if (bounds.width < aspectRatio * bounds.height)
			adjustWidth(aspectRatio);
		else if (bounds.width > aspectRatio * bounds.height)
			adjustHeight(aspectRatio);
	}

	/**
	 * Adjusts the width of the annotation according to the given aspect ratio. The height remains unchanged.
	 *
	 * @param[in] aspectRatio Aspect ratio of the window (width / height).
	 */
	void adjustWidth(double aspectRatio) {
		int cx = bounds.x + bounds.width / 2;
		bounds.width = static_cast<int>(std::round(aspectRatio * bounds.height));
		bounds.x = cx - bounds.width / 2;
	}

	/**
	 * Adjusts the height of the annotation according to the given aspect ratio. The width remains unchanged.
	 *
	 * @param[in] aspectRatio Aspect ratio of the window (width / height).
	 */
	void adjustHeight(double aspectRatio) {
		int cy = bounds.y + bounds.height / 2;
		bounds.height = static_cast<int>(std::round(bounds.width / aspectRatio));
		bounds.y = cy - bounds.height / 2;
	}

	cv::Rect bounds; ///< Bounding box.
	bool fuzzy = false; ///< Flag that indicates whether this annotation is neither positive, nor negative.
};

} /* namespace imageio */

#endif /* IMAGEIO_ANNOTATION_HPP_ */
