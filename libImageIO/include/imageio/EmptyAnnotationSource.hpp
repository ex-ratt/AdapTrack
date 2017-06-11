/*
 * EmptyAnnotationSource.hpp
 *
 *  Created on: 23.05.2013
 *      Author: poschmann
 */

#ifndef EMPTYANNOTATIONSOURCE_HPP_
#define EMPTYANNOTATIONSOURCE_HPP_

#include "imageio/AnnotationSource.hpp"

namespace imageio {

/**
 * Annotation source without any annotations.
 */
class EmptyAnnotationSource : public AnnotationSource {
public:

	void reset() {}

	bool next() {
		return true;
	}

	Annotations getAnnotations() const {
		return empty;
	}

private:

	const Annotations empty; ///< Empty annotations.
};

} /* namespace imageio */
#endif /* EMPTYANNOTATIONSOURCE_HPP_ */
