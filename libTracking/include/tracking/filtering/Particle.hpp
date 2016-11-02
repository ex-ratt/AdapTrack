/*
 * Particle.hpp
 *
 *  Created on: 01.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_PARTICLE_HPP_
#define TRACKING_FILTERING_PARTICLE_HPP_

#include "opencv2/core/core.hpp"

namespace tracking {

namespace filtering {

/**
 * Weighted particle that represents position and velocity of a tracked target.
 *
 * The position is given as a bounding box with a center coordinate and size. The box has a fixed
 * aspect ratio that is equal across all particles. The velocity (motion in between two subsequent
 * frames) is given relative to the size of the bounding box - larger bounding boxes indicate a
 * target closer to the camera and thus are assumed to have larger positional changes than targets
 * further away from the camera.
 */
class Particle {
public:

	/**
	 * Constructs a new default particle (position and velocity is zero, weight is one).
	 */
	Particle() :
			x(0), y(0), size(0), velX(0), velY(0), velSize(0), weight(1) {}

	/**
	 * Constructs a new particle with zero velocity.
	 *
	 * @param[in] x X coordinate of the bounding box center.
	 * @param[in] y Y coordinate of the bounding box center.
	 * @param[in] size Size (height) of the bounding box.
	 * @param[in] weight Importance factor.
	 */
	Particle(int x, int y, int size, double weight = 1) :
			x(x), y(y), size(size), velX(0), velY(0), velSize(0), weight(weight) {}

	/**
	 * Constructs a new particle.
	 *
	 * @param[in] x X coordinate of the bounding box center.
	 * @param[in] y Y coordinate of the bounding box center.
	 * @param[in] size Size (height) of the bounding box.
	 * @param[in] velX Velocity of the x coordinate relative to the size.
	 * @param[in] velY Velocity of the y coordinate relative to the size.
	 * @param[in] velSize Velocity of the size change relative to the size.
	 * @param[in] weight Importance factor.
	 */
	Particle(int x, int y, int size, double velX, double velY, double velSize, double weight = 1) :
			x(x), y(y), size(size), velX(velX), velY(velY), velSize(velSize), weight(weight) {}

	/**
	 * @return The bounding box.
	 */
	cv::Rect getBounds() const {
		return cv::Rect(x - getWidth() / 2, y - getHeight() / 2, getWidth(), getHeight());
	}

	/**
	 * @return X coordinate of the bounding box center.
	 */
	int getX() const {
		return x;
	}

	/**
	 * @param[in] x X coordinate of the bounding box center.
	 */
	void setX(int x) {
		this->x = x;
	}

	/**
	 * @return Y coordinate of the bounding box center.
	 */
	int getY() const {
		return y;
	}

	/**
	 * @param[in] y Y coordinate of the bounding box center.
	 */
	void setY(int y) {
		this->y = y;
	}

	/**
	 * @return Size (height) of the bounding box.
	 */
	int getSize() const {
		return size;
	}

	/**
	 * @param[in] size Size (height) of the bounding box.
	 */
	void setSize(int size) {
		this->size = size;
	}

	/**
	 * @return Height of the bounding box.
	 */
	int getHeight() const {
		return size;
	}

	/**
	 * @return Width of the bounding box.
	 */
	int getWidth() const {
		return cvRound(Particle::aspectRatio * size);
	}

	/**
	 * @return Velocity of the x coordinate relative to the size.
	 */
	double getVelX() const {
		return velX;
	}

	/**
	 * @param[in] velX Velocity of the x coordinate relative to the size.
	 */
	void setVelX(double velX) {
		this->velX = velX;
	}

	/**
	 * @return Velocity of the y coordinate relative to the size.
	 */
	double getVelY() const {
		return velY;
	}

	/**
	 * @param[in] velY Velocity of the y coordinate relative to the size.
	 */
	void setVelY(double velY) {
		this->velY = velY;
	}

	/**
	 * @return Velocity of the size change relative to the size.
	 */
	double getVelSize() const {
		return velSize;
	}

	/**
	 * @param[in] velSize Velocity of the size change relative to the size.
	 */
	void setVelSize(double velSize) {
		this->velSize = velSize;
	}

	/**
	 * @return Importance factor.
	 */
	double getWeight() const {
		return weight;
	}

	/**
	 * @param[in] weight Importance factor.
	 */
	void setWeight(double weight) {
		this->weight = weight;
	}

	/**
	 * Determines whether this particle is less than another particle using the weight. This particle is considered
	 * less than the other particle if the weight of this one is less than the weight of the other particle.
	 *
	 * @param[in] other Other particle.
	 * @return True if this particle comes before the other in a strict weak ordering, false otherwise.
	 */
	bool operator<(const Particle& other) const {
		return weight < other.weight;
	}

	/**
	 * Determines whether this particle is bigger than another particle using the weight. This particle is considered
	 * bigger than the other particle if the weight of this one is bigger than the weight of the other particle.
	 *
	 * @param[in] other Other particle.
	 * @return True if this particle comes before the other in a strict weak ordering, false otherwise.
	 */
	bool operator>(const Particle& other) const {
		return weight > other.weight;
	}

	/**
	 * Comparison function that compares particles by their weight in ascending order.
	 */
	class WeightComparisonAsc {
	public:
		bool operator()(const Particle& lhs, const Particle& rhs) {
			return lhs.weight < rhs.weight;
		}
	};

	/**
	 * Comparison function that compares particles by their weight in descending order.
	 */
	class WeightComparisonDesc {
	public:
		bool operator()(const Particle& lhs, const Particle& rhs) {
			return lhs.weight > rhs.weight;
		}
	};

	/**
	 * Changes the aspect ratio of all particles.
	 *
	 * @param[in] aspectRatio Aspect ratio (width / height) of all particles.
	 */
	static void setAspectRatio(double aspectRatio) {
		Particle::aspectRatio = aspectRatio;
	}

	/**
	 * Changes the aspect ratio of all particles to the ratio between the given width and height.
	 *
	 * @param[in] width Width relative to the given height.
	 * @param[in] height Height relative to the given width.
	 */
	static void setAspectRatio(double width, double height) {
		setAspectRatio(width / height);
	}

private:

	static double aspectRatio; ///< Aspect ratio (width / height) of all particles.

	int x; ///< X coordinate of the bounding box center.
	int y; ///< Y coordinate of the bounding box center.
	int size; ///< Size (height) of the bounding box.
	double velX; ///< Velocity of the x coordinate relative to the size.
	double velY; ///< Velocity of the y coordinate relative to the size.
	double velSize; ///< Velocity of the size change relative to the size.
	double weight; ///< Importance factor.
};

} // namespace filtering

} // namespace tracking

#endif /* TRACKING_FILTERING_PARTICLE_HPP_ */
