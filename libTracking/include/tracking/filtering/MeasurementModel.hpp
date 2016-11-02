/*
 * MeasurementModel.hpp
 *
 *  Created on: 01.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_MEASUREMENTMODEL_HPP_
#define TRACKING_FILTERING_MEASUREMENTMODEL_HPP_

#include "imageprocessing/VersionedImage.hpp"
#include "tracking/filtering/Particle.hpp"
#include <vector>
#include <memory>

namespace tracking {

namespace filtering {

/**
 * Measurement model used by particle filters.
 */
class MeasurementModel {
public:

	virtual ~MeasurementModel() {}

	/**
	 * Updates this model so all subsequent calls to evaluate use the data of the new image.
	 *
	 * @param[in] image The new image.
	 */
	virtual void update(std::shared_ptr<imageprocessing::VersionedImage> image) = 0;

	/**
	 * Changes the weight of the particle according to the likelihood of an object existing at its position in the image.
	 *
	 * @param[in] particle The particle whose weight will be changed according to the likelihood.
	 */
	virtual void evaluate(Particle& particle) const = 0;
};

} // namespace filtering

} // namespace tracking

#endif /* TRACKING_FILTERING_MEASUREMENTMODEL_HPP_ */
