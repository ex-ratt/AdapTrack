/*
 * SingleAnnotationSource.hpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#ifndef SINGLEANNOTATIONSOURCE_HPP_
#define SINGLEANNOTATIONSOURCE_HPP_

#include "imageio/AnnotationSource.hpp"
#include "opencv2/core/core.hpp"
#include <string>
#include <vector>

namespace imageio {

/**
 * Annotation source that has at most one annotation per frame. The file must have one line per frame
 * containing four values: the x and y coordinate of the upper left corner, followed by the width and
 * height. If width or height is less than one, then the target is considered invisible. The values
 * should be separated by white space or a delimiter character or both.
 */
class SingleAnnotationSource : public AnnotationSource {
public:

	/**
	 * Constructs a new single annotation source.
	 *
	 * @param[in] filename The name of the file containing the annotations.
	 */
	SingleAnnotationSource(const std::string& filename);

	void reset();

	bool next();

	Annotations getAnnotations() const;

private:

	std::vector<Annotations> annotations; ///< The annotations.
	int index;                            ///< The index of the current annotation.
};

} /* namespace imageio */
#endif /* SINGLEANNOTATIONSOURCE_HPP_ */
