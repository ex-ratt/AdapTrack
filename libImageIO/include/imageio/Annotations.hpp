/*
 * Annotations.hpp
 *
 *  Created on: Jun 9, 2017
 *      Author: ex-ratt
 */

#ifndef IMAGEIO_ANNOTATIONS_HPP_
#define IMAGEIO_ANNOTATIONS_HPP_

#include "imageio/Annotation.hpp"
#include <vector>

namespace imageio {

/**
 * Collection of rectangular annotations.
 */
struct Annotations {

	/**
	 * Adjusts the sizes of the annotations according to the given aspect ratio. If the aspect ratio of a
	 * bounding box diverges from the given one, then either the width or height is increased.
	 *
	 * @param[in] aspectRatio Aspect ratio of the window (width / height).
	 */
	void adjustSizes(double aspectRatio) {
		for (Annotation& annotation : annotations)
			annotation.adjustSize(aspectRatio);
	}

	/**
	 * Adjusts the width of the annotations according to the given aspect ratio. The height remains unchanged.
	 *
	 * @param[in] aspectRatio Aspect ratio of the window (width / height).
	 */
	void adjustWidths(double aspectRatio) {
		for (Annotation& annotation : annotations)
			annotation.adjustWidth(aspectRatio);
	}

	/**
	 * Adjusts the height of the annotations according to the given aspect ratio. The width remains unchanged.
	 *
	 * @param[in] aspectRatio Aspect ratio of the window (width / height).
	 */
	void adjustHeights(double aspectRatio) {
		for (Annotation& annotation : annotations)
			annotation.adjustHeight(aspectRatio);
	}

	size_t allCount() const {
		return annotations.size();
	}

	size_t positiveCount() const {
		return std::count_if(annotations.begin(), annotations.end(), [](auto annotation) {
			return !annotation.fuzzy;
		});
	}

	size_t fuzzyCount() const {
		return std::count_if(annotations.begin(), annotations.end(), [](auto annotation) {
			return annotation.fuzzy;
		});
	}

	/**
	 * @return All annotated bounding boxes.
	 */
	std::vector<cv::Rect> allAnnotations() const {
		std::vector<cv::Rect> boundingBoxes;
		for (Annotation annotation : annotations)
			boundingBoxes.push_back(annotation.bounds);
		return boundingBoxes;
	}

	/**
	 * @return Positive bounding boxes.
	 */
	std::vector<cv::Rect> positiveAnnotations() const {
		std::vector<cv::Rect> boundingBoxes;
		for (Annotation annotation : annotations) {
			if (!annotation.fuzzy)
				boundingBoxes.push_back(annotation.bounds);
		}
		return boundingBoxes;
	}

	/**
	 * @return Fuzzy bounding boxes (neither positive, nor negative).
	 */
	std::vector<cv::Rect> fuzzyAnnotations() const {
		std::vector<cv::Rect> boundingBoxes;
		for (Annotation annotation : annotations) {
			if (annotation.fuzzy)
				boundingBoxes.push_back(annotation.bounds);
		}
		return boundingBoxes;
	}

	std::vector<Annotation> annotations; ///< Annotations.
};

} /* namespace imageio */

#endif /* IMAGEIO_ANNOTATIONS_HPP_ */
