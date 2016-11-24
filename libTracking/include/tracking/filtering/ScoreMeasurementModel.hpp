/*
 * ScoreMeasurementModel.hpp
 *
 *  Created on: 01.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_SCOREMEASUREMENTMODEL_HPP_
#define TRACKING_FILTERING_SCOREMEASUREMENTMODEL_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include "tracking/filtering/MeasurementModel.hpp"
#include <functional>

namespace tracking {

namespace filtering {

/**
 * Measurement model that extracts a score from the target position and transforms it into a likelihood.
 */
class ScoreMeasurementModel : public MeasurementModel {
public:

	/**
	 * Constructs a new score measurement model.
	 *
	 * @param[in] scoreExtractor Extractor of the classifier score.
	 * @param[in] likelihoodFunction Function that computes likelihoods from scores.
	 */
	ScoreMeasurementModel(
			std::shared_ptr<imageprocessing::FeatureExtractor> scoreExtractor,
			std::function<double(double)> likelihoodFunction) :
		scoreExtractor(scoreExtractor),
		likelihoodFunction(likelihoodFunction) {}

	void update(std::shared_ptr<imageprocessing::VersionedImage> image) {
		scoreExtractor->update(image);
	}

	double getLikelihood(const TargetState& state) const {
		std::shared_ptr<imageprocessing::Patch> scorePatch = scoreExtractor->extract(
				state.x, state.y, state.width(), state.height());
		return scorePatch ? likelihoodFunction(scorePatch->getData().at<float>(0, 0)) : 0;
	}

private:

	std::shared_ptr<imageprocessing::FeatureExtractor> scoreExtractor; ///< Extractor of the classifier score.
	std::function<double(double)> likelihoodFunction; ///< Function that computes likelihoods from scores.
};

} // namespace filtering

} // namespace tracking

#endif /* TRACKING_FILTERING_SCOREMEASUREMENTMODEL_HPP_ */
