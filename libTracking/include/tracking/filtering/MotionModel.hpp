/*
 * MotionModel.hpp
 *
 *  Created on: 1.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_MOTIONMODEL_HPP_
#define TRACKING_FILTERING_MOTIONMODEL_HPP_

#include "tracking/filtering/TargetState.hpp"

namespace tracking {
namespace filtering {

/**
 * Motion model of a particle filter that samples new target states from previous states.
 */
class MotionModel {
public:

	virtual ~MotionModel() {}

	/**
	 * Samples a new target state.
	 *
	 * @param[in] state Target state in the previous frame.
	 * @return Sampled target state in the current frame.
	 */
	virtual TargetState sample(const TargetState& state) const = 0;
};

} // namespace filtering
} // namespace tracking

#endif /* TRACKING_FILTERING_MOTIONMODEL_HPP_ */
