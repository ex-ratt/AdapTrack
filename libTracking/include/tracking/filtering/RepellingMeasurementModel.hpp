/*
 * RepellingMeasurementModel.hpp
 *
 *  Created on: 23.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_REPELLINGMEASUREMENTMODEL_HPP_
#define TRACKING_FILTERING_REPELLINGMEASUREMENTMODEL_HPP_

#include "tracking/filtering/MeasurementModel.hpp"

namespace tracking {
namespace filtering {

/**
 * Measurement model that reduces the likelihood of particles that are close to other targets.
 */
class RepellingMeasurementModel : public MeasurementModel {
public:

	void update(std::shared_ptr<imageprocessing::VersionedImage> image) override {}

	double getLikelihood(const TargetState& state) const override {
		cv::Rect stateBounds = state.bounds();
		double likelihood = 1.0;
		for (const TargetState& target : otherTargets)
			likelihood *= std::max(0.0, 1 - 3 * computeOverlap(stateBounds, target.bounds()));
		return likelihood;
	}

	/**
	 * Sets the states of the targets that repel the particles.
	 */
	void setOtherTargets(const std::vector<TargetState>& otherTargets) {
		this->otherTargets = otherTargets;
	}

private:

	double computeOverlap(cv::Rect a, cv::Rect b) const {
		double intersectionArea = (a & b).area();
		double unionArea = a.area() + b.area() - intersectionArea;
		return intersectionArea / unionArea;
	}

	std::vector<TargetState> otherTargets; ///< Targets that repel the particles.
};

} // namespace filtering
} // namespace tracking

#endif /* TRACKING_FILTERING_REPELLINGMEASUREMENTMODEL_HPP_ */
