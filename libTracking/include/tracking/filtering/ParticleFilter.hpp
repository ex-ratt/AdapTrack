/*
 * ParticleFilter.hpp
 *
 *  Created on: 01.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_PARTICLEFILTER_HPP_
#define TRACKING_FILTERING_PARTICLEFILTER_HPP_

#include "imageprocessing/VersionedImage.hpp"
#include "opencv2/core/core.hpp"
#include "tracking/filtering/MeasurementModel.hpp"
#include "tracking/filtering/MotionModel.hpp"
#include "tracking/filtering/Particle.hpp"
#include "tracking/filtering/TargetState.hpp"
#include <memory>
#include <random>
#include <vector>

namespace tracking {
namespace filtering {

/**
 * Bootstrap filter (simple variation of a particle filter) that estimates the three-dimensional position
 * (bounding box with fixed aspect ratio) and velocity of a single target in image sequences.
 */
class ParticleFilter {
public:

	/**
	 * Constructs a new particle filter.
	 *
	 * @param[in] motionModel Motion model.
	 * @param[in] measurementModel Measurement model.
	 * @param[in] count Number of particles.
	 */
	ParticleFilter(
			std::shared_ptr<MotionModel> motionModel,
			std::shared_ptr<MeasurementModel> measurementModel,
			int count);

	/**
	 * Initializes this filter at the given position.
	 *
	 * @param[in] image Current image.
	 * @param[in] position Inital target position.
	 * @param[in] positionDeviation Standard deviation of the position relative to the size.
	 * @param[in] velocityDeviation Standard deviation of the velocity relative to the size.
	 */
	void initialize(const std::shared_ptr<imageprocessing::VersionedImage> image, const cv::Rect& position,
			double positionDeviation = 0.1, double velocityDeviation = 0.1);

	/**
	 * Determines the most probable target state within the current image.
	 *
	 * @param[in] image Current image.
	 * @return Most probable target state.
	 */
	TargetState update(const std::shared_ptr<imageprocessing::VersionedImage> image);

	/**
	 * @return Weighted particles.
	 */
	const std::vector<Particle>& getParticles() const {
		return particles;
	}

private:

	void resampleParticles();

	void moveParticles();

	void weightParticles(const std::shared_ptr<imageprocessing::VersionedImage> image);

	void normalizeParticleWeights();

	TargetState computeAverageState();

	std::default_random_engine generator; ///< Random number generator.
	std::uniform_real_distribution<> standardUniform; ///< Uniform distribution of values in [0, 1).
	mutable std::normal_distribution<> standardGaussian; ///< Normal distribution with zero mean and unit variance.
	std::shared_ptr<MotionModel> motionModel; ///< Motion model.
	std::shared_ptr<MeasurementModel> measurementModel; ///< Measurement model.
	std::vector<Particle> particles; ///< Weighted particles.
};

} // namespace filtering
} // namespace tracking

#endif /* TRACKING_FILTERING_PARTICLEFILTER_HPP_ */
