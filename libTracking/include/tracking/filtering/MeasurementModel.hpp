/*
 * MeasurementModel.hpp
 *
 *  Created on: 01.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_MEASUREMENTMODEL_HPP_
#define TRACKING_FILTERING_MEASUREMENTMODEL_HPP_

#include "imageprocessing/VersionedImage.hpp"
#include "tracking/filtering/TargetState.hpp"
#include <memory>

namespace tracking {
namespace filtering {

/**
 * Measurement model of a particle filter that computes the likelihood of target states.
 */
class MeasurementModel {
public:

	virtual ~MeasurementModel() {}

	/**
	 * Updates this model with new image data that is used to compute the state likelihoods.
	 *
	 * @param[in] image New image.
	 */
	virtual void update(std::shared_ptr<imageprocessing::VersionedImage> image) = 0;

	/**
	 * Computes the likelihood of a target state.
	 *
	 * @param[in] state Target state.
	 * @return Likelihood of the target state.
	 */
	virtual double getLikelihood(const TargetState& state) const = 0;
};

} // namespace filtering
} // namespace tracking

#endif /* TRACKING_FILTERING_MEASUREMENTMODEL_HPP_ */
