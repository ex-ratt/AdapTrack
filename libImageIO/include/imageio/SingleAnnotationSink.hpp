/*
 * SingleAnnotationSink.hpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#ifndef SINGLEANNOTATIONSINK_HPP_
#define SINGLEANNOTATIONSINK_HPP_

#include "imageio/AnnotationSink.hpp"
#include <fstream>

namespace imageio {

/**
 * Annotation sink that writes annotations to a file, one per frame and line. There will be four values:
 * the x and y coordinate of the upper left corner, followed by the width and height. If there is no
 * annotation, then all values will be zero. The values are separated by whitespace.
 */
class SingleAnnotationSink : public AnnotationSink {
public:

	/**
	 * Constructs a new single annotation sink.
	 *
	 * @param[in] precision The decimal precision of the output.
	 */
	explicit SingleAnnotationSink(size_t precision = 0);

	/**
	 * Constructs a new single annotation sink and opens a file to write into.
	 *
	 * @param[in] filename The name of the file to write the annotation data into.
	 * @param[in] precision The decimal precision of the output.
	 */
	explicit SingleAnnotationSink(const std::string& filename, size_t precision = 0);

	~SingleAnnotationSink();

	SingleAnnotationSink(SingleAnnotationSink& other) = delete;

	SingleAnnotationSink operator=(SingleAnnotationSink rhs) = delete;

	bool isOpen();

	void open(const std::string& filename);

	void close();

	void add(Annotations annotations);

private:

	std::ofstream output; ///< The file output stream.
};

} /* namespace imageio */
#endif /* SINGLEANNOTATIONSINK_HPP_ */
