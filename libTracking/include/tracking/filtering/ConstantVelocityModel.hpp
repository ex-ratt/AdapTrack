/*
 * ConstantVelocityModel.hpp
 *
 *  Created on: 01.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_CONSTANTVELOCITYMODEL_HPP_
#define TRACKING_FILTERING_CONSTANTVELOCITYMODEL_HPP_

#include "tracking/filtering/MotionModel.hpp"
#include <random>

namespace tracking {
namespace filtering {

/**
 * Motion model that assumes a constant velocity with Gaussian process noise.
 *
 * The process noise is applied to the velocity and the new position is computed with the old position
 * and new velocity.
 */
class ConstantVelocityModel : public MotionModel {
public:

	/**
	 * Constructs a new constant velocity model.
	 *
	 * @param[in] deviation Standard deviation of the process noise that is applied to the velocity.
	 */
	explicit ConstantVelocityModel(double deviation) :
			generator(std::random_device()()),
			standardGaussian(0, 1),
			deviation(deviation) {}

	TargetState sample(const TargetState& state) const override {
		double velX = state.velX + deviation * standardGaussian(generator);
		double velY = state.velY + deviation * standardGaussian(generator);
		double velSize = state.velSize + deviation * standardGaussian(generator);
		int x = static_cast<int>(std::round(state.x + velX * state.size));
		int y = static_cast<int>(std::round(state.y + velY * state.size));
		int size = static_cast<int>(std::round(state.size + velSize * state.size));
		return TargetState(x, y, size, velX, velY, velSize);
	}

private:

	mutable std::default_random_engine generator; ///< Random number generator.
	mutable std::normal_distribution<> standardGaussian; ///< Normal distribution with zero mean and unit variance.
	double deviation; ///< Standard deviation of the process noise that is applied to the velocity.
};

} // namespace filtering
} // namespace tracking

#endif /* TRACKING_FILTERING_CONSTANTVELOCITYMODEL_HPP_ */
