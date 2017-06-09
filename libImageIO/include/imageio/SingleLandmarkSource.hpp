/*
 * SingleLandmarkSource.hpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#ifndef SIMPLELANDMARKSOURCE_HPP_
#define SIMPLELANDMARKSOURCE_HPP_

#include "imageio/LandmarkCollection.hpp"
#include "imageio/LandmarkSource.hpp"
#include "opencv2/core/core.hpp"
#include <string>
#include <vector>

namespace imageio {

/**
 * Landmark source that has at most one rectangular landmark per frame named "target". The file must have one
 * line per frame containing four values: the x and y coordinate of the upper left corner, followed by the
 * width and height. If width or height is less than one, then the target is considered invisible. The values
 * should be separated by white space or a delimiter character (delimiter and whitespace is also allowed).
 */
class SingleLandmarkSource : public LandmarkSource {
public:

	/**
	 * Constructs a new simple landmark source.
	 *
	 * @param[in] filename The name of the file containing the landmark data.
	 */
	SingleLandmarkSource(const std::string& filename);

	void reset();

	bool next();

	LandmarkCollection getLandmarks() const;

private:

	static const std::string landmarkName;   ///< The name of the landmarks.
	std::vector<cv::Rect_<float>> positions; ///< The target positions.
	int index;                               ///< The index of the current target position.
};

} /* namespace imageio */
#endif /* SIMPLELANDMARKSOURCE_HPP_ */
