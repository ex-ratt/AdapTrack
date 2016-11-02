/*
 * Tracker.cpp
 *
 *  Created on: 02.11.2016
 *      Author: poschmann
 */

#include "tracking/Tracker.hpp"
#include "tracking/filtering/ScoreMeasurementModel.hpp"
#include "imageprocessing/Patch.hpp"

using boost::optional;
using classification::ProbabilisticSvmClassifier;
using cv::Mat;
using cv::Rect;
using detection::AggregatedFeaturesDetector;
using tracking::filtering::ParticleFilter;
using tracking::filtering::MeasurementModel;
using tracking::filtering::Particle;
using tracking::filtering::ScoreMeasurementModel;
using tracking::filtering::MotionModel;
using imageprocessing::extraction::AggregatedFeaturesExtractor;
using imageprocessing::VersionedImage;
using std::make_shared;
using std::shared_ptr;
using std::vector;

namespace tracking {

Tracker::Tracker(shared_ptr<AggregatedFeaturesDetector> detector,
		shared_ptr<ProbabilisticSvmClassifier> probabilisticSvm,
		shared_ptr<MotionModel> motionModel,
		int particleCount) :
				useTracker(false),
				detector(detector),
				filter(),
				scoreExtractor(),
				likelihoodFunction([&](double score) { return probabilisticSvm->getProbability(score).second; }),
				versionedImage(make_shared<VersionedImage>()),
				particleLocations() {
	const shared_ptr<AggregatedFeaturesExtractor> featureExtractor = detector->getFeatureExtractor();
	scoreExtractor = make_shared<AggregatedFeaturesExtractor>(detector->getScorePyramid(),
			featureExtractor->getPatchSizeInCells(), featureExtractor->getCellSizeInPixels(), false);
	shared_ptr<MeasurementModel> measurementModel = make_shared<ScoreMeasurementModel>(scoreExtractor, likelihoodFunction);
	filter = make_shared<ParticleFilter>(motionModel, measurementModel, particleCount);
}

void Tracker::reset() {
	useTracker = false;
}

vector<Rect> Tracker::update(const Mat& image) {
	particleLocations.clear();
	vector<Rect> objects;
	versionedImage->setData(image);
	if (useTracker) {
		optional<Rect> boundingBox = filter->update(versionedImage);
		for (const Particle& particle : filter->getParticles())
			particleLocations.emplace_back(particle.getBounds(), particle.getWeight());
		if (isValidTarget(boundingBox)) {
			objects.push_back(*boundingBox);
			useTracker = true;
		} else {
			useTracker = false;
		}
	} else {
		vector<Rect> detections = detector->detect(versionedImage);
		for (const Rect& boundingBox : detections)
			objects.push_back(boundingBox);
		if (detections.size() > 0) {
			auto max = std::max_element(detections.begin(), detections.end(), [](const Rect& a, const Rect& b) {
				return a.height < b.height;
			});
			filter->initialize(versionedImage, *max);
			useTracker = true;
		}
	}
	// TODO
	// move particles
	// update scores using new image
	// weight particles using scores
	// weight particles using other means (distance to tracks, optical flow, adaptive stuff)
	// detect heads using scores
	// pick associations between tracks and heads
	// initialize new tracks on unassigned heads (proper initialization that somehow ignores false positives)
	// somehow delete tracks
	return objects;
}

bool Tracker::isValidTarget(const optional<Rect>& boundingBox) const {
	if (!boundingBox)
		return false;
	std::shared_ptr<imageprocessing::Patch> scorePatch = scoreExtractor->extract(*boundingBox);
	if (!scorePatch)
		return false;
	double score = scorePatch->getData().at<float>(0, 0);
	double likelihood = likelihoodFunction(score);
	return likelihood > 0.05;
}

} // namespace tracking
