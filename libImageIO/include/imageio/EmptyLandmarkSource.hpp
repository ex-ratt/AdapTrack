/*
 * EmptyLandmarkSource.hpp
 *
 *  Created on: 23.05.2013
 *      Author: poschmann
 */

#ifndef EMPTYLANDMARKSOURCE_HPP_
#define EMPTYLANDMARKSOURCE_HPP_

#include "imageio/LandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"

namespace imageio {

/**
 * Landmark source without any landmarks.
 */
class EmptyLandmarkSource : public LandmarkSource {
public:

	/**
	 * Constructs a new empty landmark source.
	 */
	EmptyLandmarkSource() : empty() {}

	void reset() {}

	bool next() {
		return true;
	}

	LandmarkCollection getLandmarks() const {
		return empty;
	}

private:

	const LandmarkCollection empty; ///< Empty landmark collection.
};

} /* namespace imageio */
#endif /* EMPTYLANDMARKSOURCE_HPP_ */
