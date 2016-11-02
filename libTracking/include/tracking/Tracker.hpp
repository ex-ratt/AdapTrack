/*
 * Tracker.hpp
 *
 *  Created on: 02.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_TRACKER_HPP_
#define TRACKING_TRACKER_HPP_

#include "classification/ProbabilisticSvmClassifier.hpp"
#include "detection/AggregatedFeaturesDetector.hpp"
#include "filtering/MotionModel.hpp"
#include "filtering/ParticleFilter.hpp"
#include "imageprocessing/extraction/AggregatedFeaturesExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "opencv2/core/core.hpp"
#include <memory>
#include <vector>

namespace tracking {

/**
 * Tracker that estimates the position of detected targets in each frame.
 */
class Tracker {
public:

	/**
	 * Constructs a new tracker.
	 *
	 * TODO params
	 */
	explicit Tracker(std::shared_ptr<detection::AggregatedFeaturesDetector> detector,
			std::shared_ptr<classification::ProbabilisticSvmClassifier> probabilisticSvm,
			std::shared_ptr<filtering::MotionModel> motionModel,
			int particleCount);

	/**
	 * Detects new and tracks already detected targets.
	 *
	 * @param[in] image Current image.
	 * @return Bounding boxes around the tracked targets.
	 */
	std::vector<cv::Rect> update(const cv::Mat& image);

	/**
	 * Resets the tracker to its initial state.
	 */
	void reset();

	const std::vector<std::pair<cv::Rect, double>> getParticleLocations() const {
		return particleLocations;
	}

private:

	/**
	 * Determines whether the given bounding box is a valid target.
	 *
	 * @param[in] boundingBox Bounding box indicating the target position.
	 * @return True if the bounding box probably represents the target, false otherwise.
	 */
	bool isValidTarget(const boost::optional<cv::Rect>& boundingBox) const;

	bool useTracker; ///< Flag that indicates whether to use the tracker.
	std::shared_ptr<detection::AggregatedFeaturesDetector> detector; ///< Detector.
	std::shared_ptr<filtering::ParticleFilter> filter; ///< Particle filter.
	std::shared_ptr<imageprocessing::extraction::AggregatedFeaturesExtractor> scoreExtractor;
	std::function<double(double)> likelihoodFunction; ///< Function that computes likelihoods from scores.
	std::shared_ptr<imageprocessing::VersionedImage> versionedImage; ///< Current image and version number.

	std::vector<std::pair<cv::Rect, double>> particleLocations;
};

} // namespace tracking

#endif /* TRACKING_TRACKER_HPP_ */
