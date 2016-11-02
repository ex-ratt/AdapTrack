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
#include "imageprocessing/VersionedImage.hpp"
#include "tracking/filtering/MeasurementModel.hpp"
#include "tracking/filtering/Particle.hpp"
#include <functional>

namespace tracking {

namespace filtering {

/**
 * Measurement model that extracts a score from the sample position and transforms it into a likelihood.
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

	void evaluate(Particle& particle) const {
		std::shared_ptr<imageprocessing::Patch> scorePatch = scoreExtractor->extract(
				particle.getX(), particle.getY(), particle.getWidth(), particle.getHeight());
		if (!scorePatch) {
			particle.setWeight(0);
		} else {
			double score = scorePatch->getData().at<float>(0, 0);
			double likelihood = likelihoodFunction(score);
			particle.setWeight(particle.getWeight() * likelihood);
		}
	}

private:

	std::shared_ptr<imageprocessing::FeatureExtractor> scoreExtractor; ///< Extractor of the classifier score.
	std::function<double(double)> likelihoodFunction; ///< Function that computes likelihoods from scores.
};

} // namespace filtering

} // namespace tracking

#endif /* TRACKING_FILTERING_SCOREMEASUREMENTMODEL_HPP_ */
