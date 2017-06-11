/*
 * AnnotatedImageSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef ANNOTATEDIMAGESOURCE_HPP_
#define ANNOTATEDIMAGESOURCE_HPP_

#include "imageio/AnnotatedImage.hpp"
#include "imageio/AnnotationSource.hpp"
#include "imageio/ImageSource.hpp"

namespace imageio {

/**
 * Source of subsequent annotated images.
 */
class AnnotatedImageSource : public ImageSource, public AnnotationSource {
public:

	virtual ~AnnotatedImageSource() {}

	virtual void reset() = 0;

	virtual bool next() = 0;

	virtual const cv::Mat getImage() const = 0;

	virtual std::string getName() const = 0;

	virtual Annotations getAnnotations() const = 0;

	/**
	 * Retrieves the current annotated image.
	 *
	 * @return The annotated image (that may be empty if no data could be retrieved).
	 */
	virtual AnnotatedImage getAnnotatedImage() const {
		return AnnotatedImage{getImage(), getAnnotations()};
	}
};

} /* namespace imageio */
#endif /* ANNOTATEDIMAGESOURCE_HPP_ */
