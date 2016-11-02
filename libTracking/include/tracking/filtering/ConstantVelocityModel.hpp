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

	void sample(Particle& particle) const {
		particle.setVelX(particle.getVelX() + deviation * standardGaussian(generator));
		particle.setVelY(particle.getVelY() + deviation * standardGaussian(generator));
		particle.setVelSize(particle.getVelSize() + deviation * standardGaussian(generator));
		particle.setX(static_cast<int>(std::round(particle.getX() + particle.getVelX() * particle.getSize())));
		particle.setY(static_cast<int>(std::round(particle.getY() + particle.getVelY() * particle.getSize())));
		particle.setSize(static_cast<int>(std::round(particle.getSize() * (1 + particle.getVelSize()))));
	}

private:

	mutable std::default_random_engine generator; ///< Random number generator.
	mutable std::normal_distribution<> standardGaussian; ///< Normal distribution with zero mean and unit variance.
	double deviation; ///< Standard deviation of the process noise that is applied to the velocity.
};

} // namespace filtering

} // namespace tracking

#endif /* TRACKING_FILTERING_CONSTANTVELOCITYMODEL_HPP_ */
