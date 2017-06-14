/*	
 * ImageSource.hpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann, Patrik Huber
 */

#ifndef IMAGESOURCE_HPP_
#define IMAGESOURCE_HPP_

#include "opencv2/core/core.hpp"
#include <string>

namespace imageio {

/**
 * Source of subsequent images.
 */
class ImageSource {
public:

	virtual ~ImageSource() {}

	/**
	 * Resets this source to its initial state.
	 */
	virtual void reset() = 0;

	/**
	 * Proceeds to the next image of this source.
	 *
	 * @return True if there is an image to proceed to, false otherwise.
	 */
	virtual bool next() = 0;

	/**
	 * Retrieves the current image.
	 *
	 * @return The image (that may be empty if no data could be retrieved).
	 */
	virtual const cv::Mat getImage() const = 0;

	/**
	 * Retrieves the name of the current image.
	 *
	 * @return The name of the current image (that may be empty if no data could be retrieved).
	 */
	virtual std::string getName() const = 0;
};

} /* namespace imageio */
#endif /* IMAGESOURCE_HPP_ */
