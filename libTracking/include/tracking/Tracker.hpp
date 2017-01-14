/*
 * Tracker.hpp
 *
 *  Created on: 02.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_TRACKER_HPP_
#define TRACKING_TRACKER_HPP_

#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/IncrementalClassifierTrainer.hpp"
#include "detection/AggregatedFeaturesDetector.hpp"
#include "imageprocessing/extraction/AggregatedFeaturesExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "opencv2/core/core.hpp"
#include "tracking/filtering/MeasurementModel.hpp"
#include "tracking/filtering/MotionModel.hpp"
#include "tracking/filtering/ParticleFilter.hpp"
#include "tracking/filtering/RepellingMeasurementModel.hpp"
#include "tracking/filtering/TargetState.hpp"
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace tracking {

/**
 * Tracked target.
 */
struct Track {
	int id; ///< Unique identifier.
	std::shared_ptr<classification::ProbabilisticSvmClassifier> svm; ///< SVM that is adapted to the target.
	std::shared_ptr<classification::IncrementalClassifierTrainer<classification::ProbabilisticSvmClassifier>> svmTrainer; ///< SVM trainer.
	std::shared_ptr<filtering::RepellingMeasurementModel> repellingMeasurementModel; ///< Measurement model that penalizes close targets.
	std::unique_ptr<filtering::ParticleFilter> filter; ///< Particle filter.
	filtering::TargetState state; ///< Current state of the target.
	bool established; ///< Flag that indicates whether the track is established.
	bool visible; ///< Flag that indicates whether the target is visible.
	bool adapted; ///< Flag that indicates whether the track adapted to the target in the current frame.
	std::vector<std::pair<cv::Rect, double>> particles() const {
		std::vector<std::pair<cv::Rect, double>> particles;
		for (const filtering::Particle& particle : filter->getParticles())
			particles.emplace_back(particle.state.bounds(), particle.weight);
		return particles;
	}
};

/**
 * Data associations between tracks and detections.
 */
struct Associations {
	std::vector<std::reference_wrapper<Track>> matchedTracks; ///< Tracks with an associated detection.
	std::vector<std::reference_wrapper<Track>> unmatchedTracks; ///< Tracks without an associated detection.
	std::vector<cv::Rect> unmatchedDetections; ///< Detections without an associated track.
};

/**
 * Tracker that estimates the position of detected targets in each frame.
 */
class Tracker {
public:

	/**
	 * Constructs a new tracker.
	 *
	 * @param[in] detector Detector used to initialize the tracking.
	 * @param[in] probabilisticSvm SVM used to compute the likelihood of the particles.
	 * @param[in] motionModel Motion model used to sample new particles.
	 */
	Tracker(std::shared_ptr<detection::AggregatedFeaturesDetector> detector,
			std::shared_ptr<classification::ProbabilisticSvmClassifier> probabilisticSvm,
			std::shared_ptr<filtering::MotionModel> motionModel);

	/**
	 * Detects new and tracks already detected targets.
	 *
	 * @param[in] image Current image.
	 * @return Identifiers of and bounding boxes around the tracked targets.
	 */
	std::vector<std::pair<int, cv::Rect>> update(const cv::Mat& image);

	/**
	 * Resets the tracker to its initial state.
	 */
	void reset();

	/**
	 * @return All tracked targets.
	 */
	const std::vector<Track>& getTracks() const;

private:

	/**
	 * Updates the image data and feature pyramid.
	 *
	 * @param[in] image New image data.
	 */
	void updateImage(const cv::Mat& image);

	/**
	 * Updates the particle filters of the tracks.
	 */
	void updateFilters();

	/**
	 * Determines the associations between the tracked and detected targets.
	 *
	 * @param[in] tracks Tracked targets.
	 * @param[in] detections Detected targets.
	 * @return Associations between tracks and detections.
	 */
	Associations pickAssociations(std::vector<Track>& tracks, std::vector<cv::Rect>& detections) const;

	/**
	 * Determines the similarity in position between a track and a detection by computing the overlap of the bounding boxes.
	 *
	 * @param[in] track Tracked target.
	 * @param[in] detection Detected target.
	 * @return Positional similarity between track and detection.
	 */
	double computeSimilarity(const Track& track, cv::Rect detection) const;

	/**
	 * Determines the best match between unmatched tracks and unmatched detections based on the given similarities.
	 *
	 * @param[in] similarities Matrix of similarities between tracks (rows) and detections (columns).
	 * @param[in] threshold Threshold which must be exceeded by the similarity to indicate valid matches.
	 * @param[in] unmatchedTrackIndices Indices of tracks that have not been matched with a detection yet.
	 * @param[in] unmatchedDetectionIndices Indices of detections that have not been matched with a track yet.
	 * @return Indices of the best match (y is track index, x is detection index) or (-1, -1) if there is no valid match.
	 */
	cv::Point getBestMatch(const cv::Mat& similarities, float threshold,
			const std::vector<int>& unmatchedTrackIndices, const std::vector<int>& unmatchedDetectionIndices) const;

	/**
	 * Confirms tracks with an associated detection.
	 *
	 * @param[in] matchedTracks Tracks that have an associated detection.
	 */
	void confirmMatchedTracks(std::vector<std::reference_wrapper<Track>>& matchedTracks);

	/**
	 * Removes tracks that are invalid (caused by occlusion, disappearance, or a sudden change in appearance).
	 *
	 * @param[in] unmatchedTracks Tracks that do not have an associated detection.
	 */
	void removeObsoleteTracks(std::vector<std::reference_wrapper<Track>>& unmatchedTracks);

	/**
	 * Determines whether a track is visible according to common and target specific classifier.
	 *
	 * @param[in] track Track whose visibility is determined.
	 * @return True if the track is considered visible, false otherwise.
	 */
	bool isVisible(const Track& track) const;

	/**
	 * Adds new tracks at detections without an associated track.
	 *
	 * @param[in] unmatchedDetections Detections that do not have an associated track.
	 */
	void addNewTracks(const std::vector<cv::Rect>& unmatchedDetections);

	/**
	 * Creates a new track around the given target position.
	 *
	 * @param[in] target Bounding box indicating the target position.
	 * @return Newly created track.
	 */
	Track createTrack(cv::Rect target);

	/**
	 * Updates the target specific classifiers of the tracks if reasonable.
	 */
	void updateTargetModels();

	/**
	 * Determines whether adapting a track to the current target appearance is reasonable.
	 *
	 * @param[in] track Track that may be adapted to the current target appearance.
	 * @return True if the adaptation is reasonable, false otherwise.
	 */
	bool isAdaptingReasonable(const Track& track) const;

	/**
	 * Retrieves the common SVM score of a tracked target.
	 *
	 * @param[in] state State of the target.
	 * @return Common SVM score or -1000 if (partially) outside the image.
	 */
	double getCommonScore(const filtering::TargetState& state) const;

	/**
	 * Retrieves the target-specific SVM score of a tracked target.
	 *
	 * @param[in] track Tracked target.
	 * @return Target-specific SVM score or -1000 if (partially) outside the image.
	 */
	double getTargetScore(const Track& track) const;

	/**
	 * Adapts the target-specific classifier to the current appearance of the target and its surroundings.
	 *
	 * @param[in] track Tracked target which classifier is adapted.
	 */
	void adapt(Track& track);

	/**
	 * Retrieves a single positive training example from the target position.
	 *
	 * @param[in] target Bounding box indicating the target position.
	 * @return One new positive training examples.
	 */
	std::vector<cv::Mat> getPositiveTrainingExamples(cv::Rect target) const;

	/**
	 * Retrieves hard negative training examples from the surroundings of a target (having a score above -1).
	 *
	 * @param[in] target Bounding box indicating the target position.
	 * @param[in] svm Current target-specific classifier.
	 * @return New negative training examples.
	 */
	std::vector<cv::Mat> getNegativeTrainingExamples(cv::Rect target, const classification::SvmClassifier& svm) const;

	/**
	 * Computes the overlap ratio (intersection over union) of two bounding boxes.
	 *
	 * @param[in] a A bounding box.
	 * @param[in] b Another bounding box.
	 * @return Overlap ratio of the bounding boxes.
	 */
	double computeOverlap(cv::Rect a, cv::Rect b) const;

	/**
	 * Updates the measurement models that penalize close targets with the current target positions.
	 */
	void updateRepellingModels();

	/**
	 * Extracts the IDs and bounding boxes of the tracks.
	 *
	 * @return IDs of the targets and bounding boxes indicating the positions.
	 */
	std::vector<std::pair<int, cv::Rect>> extractTargets() const;

	mutable std::default_random_engine generator; ///< Random number generator.
	std::shared_ptr<detection::AggregatedFeaturesDetector> detector; ///< Detector.
	std::shared_ptr<imageprocessing::extraction::AggregatedFeaturesExtractor> featureExtractor; ///< Feature extractor.
	std::shared_ptr<imageprocessing::extraction::AggregatedFeaturesExtractor> scoreExtractor; ///< Extractor of the SVM score common to all targets.
	std::function<double(double)> likelihoodFunction; ///< Function that computes likelihoods from scores.
	std::shared_ptr<filtering::MeasurementModel> commonMeasurementModel; ///< Measurement model common to all targets.
	std::shared_ptr<filtering::MotionModel> motionModel; ///< Motion model of the targets.

public:

	int particleCount; ///< Number of particles per target.
	bool adaptive; ///< Flag that indicates whether the tracker is adapting to the targets.
	double associationThreshold; ///< Bounding box overlap ratio that must be exceeded to match a track to a detection.
	double commonVisibilityThreshold; ///< Score that must be exceeded to consider a target visible according to the common classifier.
	double targetVisibilityThreshold; ///< Score that must be exceeded to consider a target visible according to its specific classifier.
	double commonAdaptationThreshold; ///< Score that must be exceeded to allow target adaptation according to the common classifier.
	double targetAdaptationThreshold; ///< Score that must be exceeded to allow target adaptation according to the specific classifier.
	double targetSvmC; ///< Penalty multiplier C used for training the target specific SVMs.
	double negativeOverlapThreshold; ///< Maximum allowed bounding box overlap ratio between negative training examples and target position.
	int positiveExampleCount; ///< Maximum number of positive training examples.
	int negativeExampleCount; ///< Maximum number of negative training examples.

private:

	std::shared_ptr<imageprocessing::VersionedImage> versionedImage; ///< Current image and version number.
	std::vector<Track> tracks; ///< Tracked targets.
	int nextTrackId; ///< Identifier that is not used by any track.
};

} // namespace tracking

#endif /* TRACKING_TRACKER_HPP_ */
