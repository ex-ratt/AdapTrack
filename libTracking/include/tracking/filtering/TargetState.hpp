/*
 * TargetState.hpp
 *
 *  Created on: 03.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_TARGETSTATE_HPP_
#define TRACKING_FILTERING_TARGETSTATE_HPP_

#include "opencv2/core/core.hpp"

namespace tracking {
namespace filtering {

/**
 * State of a tracked target consisting of position and velocity.
 *
 * The position is given as a bounding box with a center coordinate and size. The aspect ratio of
 * the bounding box is fixed and does not change. The velocity (motion in between two subsequent
 * frames) is given relative to the size of the bounding box - larger bounding boxes indicate a
 * target closer to the camera and thus are assumed to have larger positional changes than targets
 * further away from the camera.
 */
class TargetState {
public:

	/**
	 * Constructs a new default target state (position and velocity are zero).
	 */
	TargetState() :
			x(0), y(0), size(0), velX(0), velY(0), velSize(0) {}

	/**
	 * Constructs a new target state from bounds (ignoring the width) with zero velocity.
	 *
	 * @param[in] bounds Bounding box indicating the position.
	 */
	TargetState(cv::Rect bounds) :
			x(bounds.x + bounds.width / 2), y(bounds.y + bounds.height / 2), size(bounds.height), velX(0), velY(0), velSize(0) {}

	/**
	 * Constructs a new target state with zero velocity.
	 *
	 * @param[in] x X coordinate of the bounding box center.
	 * @param[in] y Y coordinate of the bounding box center.
	 * @param[in] size Size (height) of the bounding box.
	 */
	TargetState(int x, int y, int size) :
			x(x), y(y), size(size), velX(0), velY(0), velSize(0) {}

	/**
	 * Constructs a new target state.
	 *
	 * @param[in] x X coordinate of the bounding box center.
	 * @param[in] y Y coordinate of the bounding box center.
	 * @param[in] size Size (height) of the bounding box.
	 * @param[in] velX Velocity of the x coordinate relative to the size.
	 * @param[in] velY Velocity of the y coordinate relative to the size.
	 * @param[in] velSize Velocity of the size change relative to the size.
	 */
	TargetState(int x, int y, int size, double velX, double velY, double velSize) :
			x(x), y(y), size(size), velX(velX), velY(velY), velSize(velSize) {}

	/**
	 * @return The bounding box.
	 */
	cv::Rect bounds() const {
		return cv::Rect(x - width() / 2, y - height() / 2, width(), height());
	}

	/**
	 * @return Height of the bounding box.
	 */
	int height() const {
		return size;
	}

	/**
	 * @return Width of the bounding box.
	 */
	int width() const {
		return static_cast<int>(std::round(TargetState::aspectRatio * size));
	}

	/**
	 * Changes the aspect ratio of all target states (existing and future ones).
	 *
	 * @param[in] aspectRatio Aspect ratio (width / height).
	 */
	static void setAspectRatio(double aspectRatio) {
		TargetState::aspectRatio = aspectRatio;
	}

	/**
	 * Changes the aspect ratio of all target states (existing and future ones) to the ratio between
	 * the given width and height.
	 *
	 * @param[in] width Width relative to the given height.
	 * @param[in] height Height relative to the given width.
	 */
	static void setAspectRatio(double width, double height) {
		setAspectRatio(width / height);
	}

	int x; ///< X coordinate of the bounding box center.
	int y; ///< Y coordinate of the bounding box center.
	int size; ///< Size (height) of the bounding box.
	double velX; ///< Velocity of the x coordinate relative to the size.
	double velY; ///< Velocity of the y coordinate relative to the size.
	double velSize; ///< Velocity of the size change relative to the size.

private:

	static double aspectRatio; ///< Aspect ratio (width / height) of all target states.
};

} // namespace filtering
} // namespace tracking

#endif /* TRACKING_FILTERING_TARGETSTATE_HPP_ */
