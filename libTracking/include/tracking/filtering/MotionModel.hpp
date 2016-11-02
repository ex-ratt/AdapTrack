/*
 * MotionModel.hpp
 *
 *  Created on: 1.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_MOTIONMODEL_HPP_
#define TRACKING_FILTERING_MOTIONMODEL_HPP_

#include "tracking/filtering/Particle.hpp"

namespace tracking {

namespace filtering {

/**
 * Motion model used by particle filters.
 */
class MotionModel {
public:

	virtual ~MotionModel() {}

	/**
	 * Sample a new state for a particle.
	 *
	 * @param[in,out] particle The particle.
	 */
	virtual void sample(Particle& particle) const = 0;
};

} // namespace filtering

} // namespace tracking

#endif /* TRACKING_FILTERING_MOTIONMODEL_HPP_ */
