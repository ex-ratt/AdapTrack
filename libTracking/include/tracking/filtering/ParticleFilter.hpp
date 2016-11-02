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
#include <memory>
#include <random>
#include <vector>

namespace tracking {

namespace filtering {

/**
 * Particle filter that estimates the three-dimensional position (bounding box with fixed aspect ratio)
 * and velocity of a single target in image sequences.
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
	 * Initializes this tracker at the given position.
	 *
	 * @param[in] image Current image.
	 * @param[in] position Inital target position.
	 * @param[in] positionDeviation Standard deviation of the position relative to the size.
	 * @param[in] velocityDeviation Standard deviation of the velocity relative to the size.
	 */
	void initialize(const std::shared_ptr<imageprocessing::VersionedImage> image, const cv::Rect& position,
			double positionDeviation = 0.1, double velocityDeviation = 0.1);

	/**
	 * Updates the particles and determines the target position within the image.
	 *
	 * @param[in] image Current image.
	 * @return Bounding box around the target.
	 */
	cv::Rect update(const std::shared_ptr<imageprocessing::VersionedImage> image);

	/**
	 * @return Particles.
	 */
	const std::vector<Particle>& getParticles() const {
		return particles;
	}

private:

	void resampleParticles();

	void moveParticles();

	void weightParticles(const std::shared_ptr<imageprocessing::VersionedImage> image);

	void normalizeParticleWeights();

	cv::Rect computeAverageBounds();

	std::default_random_engine generator; ///< Random number generator.
	std::uniform_real_distribution<> standardUniform; ///< Uniform distribution of values in [0, 1).
	mutable std::normal_distribution<> standardGaussian; ///< Normal distribution with zero mean and unit variance.
	std::shared_ptr<MotionModel> motionModel; ///< Motion model.
	std::shared_ptr<MeasurementModel> measurementModel; ///< Measurement model.
	std::vector<Particle> particles; ///< Particles.
};

} // namespace filtering

} // namespace tracking

#endif /* TRACKING_FILTERING_PARTICLEFILTER_HPP_ */
