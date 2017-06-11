/*
 * AnnotationSink.hpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#ifndef ANNOTATIONSINK_HPP_
#define ANNOTATIONSINK_HPP_

#include "imageio/Annotations.hpp"
#include <string>

namespace imageio {

/**
 * Sink for subsequent annotations.
 */
class AnnotationSink {
public:

	virtual ~AnnotationSink() {}

	/**
	 * Determines whether this sink is open.
	 *
	 * @return True if this landmark sink was opened (and not closed since), false otherwise.
	 */
	virtual bool isOpen() = 0;

	/**
	 * Opens the file writer. Needs to be called before adding the first landmark collection.
	 *
	 * @param[in] filename The name of the file to write the landmark data into.
	 */
	virtual void open(const std::string& filename) = 0;

	/**
	 * Closes the file writer.
	 */
	virtual void close() = 0;

	/**
	 * Adds annotations of a frame.
	 *
	 * @param[in] annotations The annotations.
	 */
	virtual void add(Annotations annotations) = 0;
};

} /* namespace imageio */
#endif /* ANNOTATIONSINK_HPP_ */
