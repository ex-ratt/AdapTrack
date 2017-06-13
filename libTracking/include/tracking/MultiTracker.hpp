/*
 * MultiTracker.hpp
 *
 *  Created on: 02.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_MULTITRACKER_HPP_
#define TRACKING_MULTITRACKER_HPP_

#include "classification/IncrementalClassifierTrainer.hpp"
#include "classification/ProbabilisticSupportVectorMachine.hpp"
#include "detection/AggregatedFeaturesDetector.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/extraction/FeatureExtractor.hpp"
#include "opencv2/core/core.hpp"
#include "tracking/filtering/MeasurementModel.hpp"
#include "tracking/filtering/MotionModel.hpp"
#include "tracking/filtering/ParticleFilter.hpp"
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
	std::shared_ptr<classification::ProbabilisticSupportVectorMachine> svm; ///< SVM that is adapted to the target.
	std::shared_ptr<classification::IncrementalClassifierTrainer<classification::ProbabilisticSupportVectorMachine>> svmTrainer; ///< SVM trainer.
	std::unique_ptr<filtering::ParticleFilter> filter; ///< Particle filter.
	filtering::TargetState state; ///< Current state of the target.
	bool confirmed; ///< Flag that indicates whether the track was confirmed by a second detection.
	cv::Mat features; ///< Current features.
	double score; ///< SVM score of current features.
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
 * Tracker that estimates the position of multiple detected targets in each frame.
 */
class MultiTracker {
public:

	/**
	 * Constructs a new multi-target tracker.
	 *
	 * @param[in] exactFeatureExtractor Feature extractor that provides patches exactly as requested.
	 * @param[in] detector Detector that finds new targets to track.
	 * @param[in] svm SVM that computes the likelihood of the particles.
	 * @param[in] motionModel Motion model that samples new particles.
	 */
	MultiTracker(std::shared_ptr<imageprocessing::extraction::FeatureExtractor> exactFeatureExtractor,
			std::shared_ptr<detection::AggregatedFeaturesDetector> detector,
			std::shared_ptr<classification::ProbabilisticSupportVectorMachine> svm,
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
	 * Updates the image data and feature extractors.
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
	 * Determines the best match between unmatched tracks and unmatched detections based on the given overlap ratios.
	 *
	 * @param[in] overlaps Matrix of overlap ratios between tracks (rows) and detections (columns).
	 * @param[in] threshold Threshold which must be exceeded by the overlap ratio to indicate a valid match.
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
	 * Determines whether a track is visible according to the classifier score.
	 *
	 * @param[in] track Track whose visibility is determined.
	 * @return True if the track is considered visible, false otherwise.
	 */
	bool isVisible(const Track& track) const;

	/**
	 * Removes tracks that overlap with other tracks and have a lower score, thereby preventing one target to be
	 * tracked twice or more.
	 */
	void removeOverlappingTracks();

	/**
	 * Adds new tracks at detections without an associated track.
	 *
	 * @param[in] unmatchedDetections Detections that do not have an associated track.
	 */
	void addNewTracks(const std::vector<cv::Rect>& unmatchedDetections);

	/**
	 * Creates a new track at the given target position.
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
	 * Adapts the target-specific classifier to the current appearance of the target and its surroundings.
	 *
	 * @param[in] track Tracked target which classifier is adapted.
	 */
	void adapt(Track& track);

	/**
	 * Retrieves random negative training examples from the surroundings of the target.
	 *
	 * @param[in] target Bounding box indicating the target position.
	 * @return Negative training examples.
	 */
	std::vector<cv::Mat> getNegativeTrainingExamples(cv::Rect target) const;

	/**
	 * Retrieves hard negative training examples from the surroundings of the target.
	 *
	 * @param[in] target Bounding box indicating the target position.
	 * @param[in] svm Current support vector machine.
	 * @return Negative training examples.
	 */
	std::vector<cv::Mat> getNegativeTrainingExamples(cv::Rect target, const classification::SupportVectorMachine& svm) const;

	/**
	 * Computes the overlap ratio (intersection over union) of two bounding boxes.
	 *
	 * @param[in] a First bounding box.
	 * @param[in] b Second bounding box.
	 * @return Overlap ratio of the bounding boxes.
	 */
	double computeOverlap(cv::Rect a, cv::Rect b) const;

	/**
	 * Extracts the IDs and bounding boxes of the tracks.
	 *
	 * @return IDs of the targets and bounding boxes indicating the positions.
	 */
	std::vector<std::pair<int, cv::Rect>> extractTargets() const;

	mutable std::default_random_engine generator; ///< Random number generator.
	std::shared_ptr<imageprocessing::VersionedImage> versionedImage; ///< Current image and version number.
	std::vector<Track> tracks; ///< Tracked targets.
	int nextTrackId; ///< Identifier that is associated to the next new target.
	std::shared_ptr<detection::AggregatedFeaturesDetector> detector; ///< Detector that finds new targets to track.
	std::shared_ptr<imageprocessing::extraction::FeatureExtractor> pyramidFeatureExtractor; ///< Feature extractor that re-uses the feature pyramid of the detector.
	std::shared_ptr<imageprocessing::extraction::FeatureExtractor> exactFeatureExtractor; ///< Feature extractor that provides patches exactly as requested.
	std::shared_ptr<classification::ProbabilisticSupportVectorMachine> svm; ///< SVM that is common to all targets.
	std::shared_ptr<filtering::MeasurementModel> commonMeasurementModel; ///< Measurement model that is common to all targets.
	std::shared_ptr<filtering::MotionModel> motionModel; ///< Motion model of the targets.

public:

	int particleCount; ///< Number of particles per target.
	bool adaptive; ///< Flag that indicates whether the tracker is adapting to the targets.
	double associationThreshold; ///< Bounding box overlap ratio that must be exceeded to match a track to a detection.
	double visibilityThreshold; ///< Score that must be exceeded to consider a target visible.
	int negativeExampleCount; ///< Number of negative training examples per classifier update.
	double negativeOverlapThreshold; ///< Maximum allowed bounding box overlap ratio between negative training examples and target position.
	double targetSvmC; ///< Penalty multiplier C used for training the target specific SVMs.
	double learnRate; ///< Learn rate of the incremental classifier update.
};

} // namespace tracking

#endif /* TRACKING_MULTITRACKER_HPP_ */
