/*
 *
 * RandomWalkModel.hpp
 *  Created on: Jan 17, 2017
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_RANDOMWALKMODEL_HPP_
#define TRACKING_FILTERING_RANDOMWALKMODEL_HPP_

#include "tracking/filtering/MotionModel.hpp"
#include <random>
#include <stdexcept>

namespace tracking {
namespace filtering {

/**
 * Motion model that assumes a smooth change in position, where every direction of movement is equally likely.
 *
 * The process noise is Gaussian, the velocities will be computed from the sampled displacement.
 */
class RandomWalkModel : public MotionModel {
public:

	/**
	 * Constructs a new random walk model with the same standard deviation for position and size.
	 *
	 * @param[in] deviation Standard deviation of position and size relative to the size.
	 */
	explicit RandomWalkModel(double deviation) : RandomWalkModel(deviation, deviation) {}

	/**
	 * Constructs a new random walk model with different standard deviations for position and size.
	 *
	 * @param[in] positionDeviation Standard deviation of the position noise relative to the size.
	 * @param[in] sizeDeviation Standard deviation of the size noise relative to the size.
	 */
	explicit RandomWalkModel(double positionDeviation, double sizeDeviation) :
			generator(std::random_device()()),
			standardGaussian(0, 1),
			positionDeviation(positionDeviation),
			sizeDeviation(sizeDeviation) {
		if (positionDeviation <= 0.0 || sizeDeviation <= 0.0)
			throw new std::invalid_argument("RandomWalkModel: the standard deviations must be bigger than zero");
	}

	TargetState sample(const TargetState& state) const override {
		int x = static_cast<int>(std::round(state.x + positionDeviation * standardGaussian(generator) * state.size));
		int y = static_cast<int>(std::round(state.y + positionDeviation * standardGaussian(generator) * state.size));
		int size = static_cast<int>(std::round(state.size + sizeDeviation * standardGaussian(generator) * state.size));
		double velX = static_cast<double>(x - state.x) / size;
		double velY = static_cast<double>(y - state.y) / size;
		double velSize = static_cast<double>(size - state.size) / size;
		return TargetState(x, y, size, velX, velY, velSize);
	}

private:

	mutable std::default_random_engine generator; ///< Random number generator.
	mutable std::normal_distribution<> standardGaussian; ///< Normal distribution with zero mean and unit variance.
	double positionDeviation; ///< Standard deviation of the position noise relative to the size.
	double sizeDeviation; ///< Standard deviation of the size noise relative to the size.
};

} // namespace filtering
} // namespace tracking

#endif /* TRACKING_FILTERING_RANDOMWALKMODEL_HPP_ */
