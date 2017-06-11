/*
 * OrderedAnnotatedImageSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef ORDEREDANNOTATEDIMAGESOURCE_HPP_
#define ORDEREDANNOTATEDIMAGESOURCE_HPP_

#include "imageio/AnnotatedImageSource.hpp"
#include <memory>

namespace imageio {

/**
 * Annotated image source whose annotations are associated to the images by their order.
 */
class OrderedAnnotatedImageSource : public AnnotatedImageSource {
public:

	/**
	 * Constructs a new ordered annotated image source.
	 *
	 * @param[in] imageSource The source of the images.
	 * @param[in] annotationSource The source of the annotations.
	 */
	OrderedAnnotatedImageSource(std::shared_ptr<ImageSource> imageSource, std::shared_ptr<AnnotationSource> annotationSource);

	void reset();

	bool next();

	const cv::Mat getImage() const;

	std::string getName() const;

	Annotations getAnnotations() const;

private:

	std::shared_ptr<ImageSource> imageSource; ///< The source of the images.
	std::shared_ptr<AnnotationSource> annotationSource; ///< The source of the annotations.
};

} /* namespace imageio */
#endif /* ORDEREDANNOTATEDIMAGESOURCE_HPP_ */
