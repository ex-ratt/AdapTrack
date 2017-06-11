/*
 * AnnotationSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef ANNOTATIONSOURCE_HPP_
#define ANNOTATIONSOURCE_HPP_

#include "imageio/Annotations.hpp"

namespace imageio {

/**
 * Source of subsequent annotations.
 */
class AnnotationSource {
public:

	virtual ~AnnotationSource() {}

	/**
	 * Resets the source to its initial state.
	 */
	virtual void reset() = 0;

	/**
	 * Proceeds to the next annotations.
	 *
	 * @return True if there there are annotations to proceed to, false otherwise.
	 */
	virtual bool next() = 0;

	/**
	 * Retrieves the current annotations.
	 *
	 * @return The annotations.
	 */
	virtual Annotations getAnnotations() const = 0;
};

} /* namespace imageio */
#endif /* ANNOTATIONSOURCE_HPP_ */
