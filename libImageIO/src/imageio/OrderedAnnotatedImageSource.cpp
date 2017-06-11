/*
 * OrderedAnnotatedImageSource.cpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#include "imageio/OrderedAnnotatedImageSource.hpp"

using cv::Mat;
using std::shared_ptr;
using std::string;

namespace imageio {

OrderedAnnotatedImageSource::OrderedAnnotatedImageSource(
		shared_ptr<ImageSource> imageSource, shared_ptr<AnnotationSource> annotationSource) :
				imageSource(imageSource), annotationSource(annotationSource) {}

void OrderedAnnotatedImageSource::reset() {
	imageSource->reset();
	annotationSource->reset();
}

bool OrderedAnnotatedImageSource::next() {
	annotationSource->next();
	return imageSource->next();
}

const Mat OrderedAnnotatedImageSource::getImage() const {
	return imageSource->getImage();
}

string OrderedAnnotatedImageSource::getName() const {
	return imageSource->getName();
}

Annotations OrderedAnnotatedImageSource::getAnnotations() const {
	return annotationSource->getAnnotations();
}

} /* namespace imageio */
