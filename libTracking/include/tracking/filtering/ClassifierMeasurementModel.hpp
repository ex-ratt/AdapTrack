/*
 * ClassifierMeasurementModel.hpp
 *
 *  Created on: 02.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_CLASSIFIERMEASUREMENTMODEL_HPP_
#define TRACKING_FILTERING_CLASSIFIERMEASUREMENTMODEL_HPP_

#include "classification/ProbabilisticClassifier.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include "tracking/filtering/MeasurementModel.hpp"

namespace tracking {
namespace filtering {

/**
 * Measurement model that applies a probabilistic classifier to features extracted from the target position.
 */
class ClassifierMeasurementModel : public MeasurementModel {
public:

	/**
	 * Constructs a new classifier measurement model.
	 *
	 * @param[in] featureExtractor Extractor of features given a bounding box.
	 * @param[in] classifier Classifier that computes a probability given features.
	 */
	ClassifierMeasurementModel(
			std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor,
			std::shared_ptr<classification::ProbabilisticClassifier> classifier) :
					featureExtractor(featureExtractor),
					classifier(classifier) {}

	void update(std::shared_ptr<imageprocessing::VersionedImage> image) override {
		featureExtractor->update(image);
	}

	double getLikelihood(const TargetState& state) const override {
		std::shared_ptr<imageprocessing::Patch> featurePatch = featureExtractor->extract(
				state.x, state.y, state.width(), state.height());
		return featurePatch ? classifier->getProbability(featurePatch->getData()).second : 0;
	}

private:

	std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor; ///< Extractor of features given a bounding box.
	std::shared_ptr<classification::ProbabilisticClassifier> classifier; ///< Classifier that computes a probability given features.
};

} // namespace filtering
} // namespace tracking

#endif /* TRACKING_FILTERING_CLASSIFIERMEASUREMENTMODEL_HPP_ */
