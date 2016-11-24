/*
 * Tracker.cpp
 *
 *  Created on: 02.11.2016
 *      Author: poschmann
 */

#include "classification/AgeBasedExampleManagement.hpp"
#include "classification/FixedTrainableProbabilisticSvmClassifier.hpp"
#include "classification/LinearKernel.hpp"
#include "libsvm/LibSvmClassifier.hpp"
#include "tracking/Tracker.hpp"
#include "tracking/filtering/ClassifierMeasurementModel.hpp"
#include "tracking/filtering/CombinedMeasurementModel.hpp"
#include "tracking/filtering/ScoreMeasurementModel.hpp"
#include "imageprocessing/Patch.hpp"

using boost::optional;
using classification::AgeBasedExampleManagement;
using classification::FixedTrainableProbabilisticSvmClassifier;
using classification::LinearKernel;
using classification::ProbabilisticSvmClassifier;
using classification::SvmClassifier;
using classification::TrainableProbabilisticSvmClassifier;
using cv::Mat;
using cv::Point;
using cv::Rect;
using detection::AggregatedFeaturesDetector;
using libsvm::LibSvmClassifier;
using tracking::filtering::ClassifierMeasurementModel;
using tracking::filtering::CombinedMeasurementModel;
using tracking::filtering::MeasurementModel;
using tracking::filtering::MotionModel;
using tracking::filtering::Particle;
using tracking::filtering::ParticleFilter;
using tracking::filtering::RepellingMeasurementModel;
using tracking::filtering::ScoreMeasurementModel;
using tracking::filtering::TargetState;
using imageprocessing::Patch;
using imageprocessing::VersionedImage;
using imageprocessing::extraction::AggregatedFeaturesExtractor;
using std::make_shared;
using std::make_unique;
using std::pair;
using std::reference_wrapper;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace tracking {

Tracker::Tracker(shared_ptr<AggregatedFeaturesDetector> detector,
		shared_ptr<ProbabilisticSvmClassifier> probabilisticSvm,
		shared_ptr<MotionModel> motionModel) :
				generator(std::random_device()()),
				detector(detector),
				featureExtractor(detector->getFeatureExtractor()),
				scoreExtractor(),
				likelihoodFunction([=](double score) { return probabilisticSvm->getProbability(score).second; }),
				commonMeasurementModel(),
				motionModel(motionModel),
				particleCount(500),
				adaptive(true),
				associationThreshold(0.5),
				commonVisibilityThreshold(-1),
				targetVisibilityThreshold(-1),
				commonAdaptationThreshold(0.0),
				targetAdaptationThreshold(0.0),
				targetSvmC(10),
				negativeOverlapThreshold(0.5),
				positiveExampleCount(1),
				negativeExampleCount(20),
				versionedImage(make_shared<VersionedImage>()),
				tracks(),
				nextTrackId(0) {
	scoreExtractor = make_shared<AggregatedFeaturesExtractor>(detector->getScorePyramid(),
			featureExtractor->getPatchSizeInCells(), featureExtractor->getCellSizeInPixels(), false);
	commonMeasurementModel = make_shared<ScoreMeasurementModel>(scoreExtractor, likelihoodFunction);
}

void Tracker::reset() {
	tracks.clear();
}

const vector<Track>& Tracker::getTracks() const {
	return tracks;
}

vector<pair<int, Rect>> Tracker::update(const Mat& image) {
	updateImage(image);
	updateFilters();
	vector<Rect> detections = detector->detect(versionedImage);
	Associations associations = pickAssociations(tracks, detections);
	confirmMatchedTracks(associations.matchedTracks);
	removeObsoleteTracks(associations.unmatchedTracks);
	addNewTracks(associations.unmatchedDetections);
	if (adaptive)
		updateTargetModels();
	updateRepellingModels();
	return extractTargets();
}

void Tracker::updateImage(const Mat& image) {
	versionedImage->setData(image);
	featureExtractor->update(versionedImage);
}

void Tracker::updateFilters() {
	for (Track& track : tracks)
		track.state = track.filter->update(versionedImage);
}

Associations Tracker::pickAssociations(vector<Track>& tracks, vector<Rect>& detections) const {
	Associations associations;
	Mat similarities(tracks.size(), detections.size(), CV_32FC1);
	for (int i = 0; i < tracks.size(); ++i) {
		for (int k = 0; k < detections.size(); ++k) {
			similarities.at<float>(i, k) = computeSimilarity(tracks[i], detections[k]);
		}
	}
	vector<int> unmatchedTrackIndices(tracks.size());
	vector<int> unmatchedDetectionIndices(detections.size());
	std::iota(unmatchedTrackIndices.begin(), unmatchedTrackIndices.end(), 0);
	std::iota(unmatchedDetectionIndices.begin(), unmatchedDetectionIndices.end(), 0);
	cv::Point nextMatch = getBestMatch(similarities, associationThreshold, unmatchedTrackIndices, unmatchedDetectionIndices);
	while (nextMatch.x >= 0) {
		associations.matchedTracks.push_back(std::ref(tracks[nextMatch.y]));
		unmatchedTrackIndices.erase(std::find(unmatchedTrackIndices.begin(), unmatchedTrackIndices.end(), nextMatch.y));
		unmatchedDetectionIndices.erase(std::find(unmatchedDetectionIndices.begin(), unmatchedDetectionIndices.end(), nextMatch.x));
		nextMatch = getBestMatch(similarities, associationThreshold, unmatchedTrackIndices, unmatchedDetectionIndices);
	}
	for (int i : unmatchedTrackIndices)
		associations.unmatchedTracks.push_back(std::ref(tracks[i]));
	for (int k : unmatchedDetectionIndices)
		associations.unmatchedDetections.push_back(detections[k]);
	return associations;
}

double Tracker::computeSimilarity(const Track& track, Rect detection) const {
	return computeOverlap(track.state.bounds(), detection);
}

Point Tracker::getBestMatch(const Mat& similarities, float threshold,
		const vector<int>& unmatchedTrackIndices, const vector<int>& unmatchedDetectionIndices) const {
	Point maxElement(-1, -1);
	float maxSimilarity = threshold;
	for (int trackIndex : unmatchedTrackIndices) {
		for (int detectionIndex : unmatchedDetectionIndices) {
			if (similarities.at<float>(trackIndex, detectionIndex) > maxSimilarity) {
				maxSimilarity = similarities.at<float>(trackIndex, detectionIndex);
				maxElement.y = trackIndex;
				maxElement.x = detectionIndex;
			}
		}
	}
	return maxElement;
}

void Tracker::confirmMatchedTracks(vector<reference_wrapper<Track>>& matchedTracks) {
	for (Track& track : matchedTracks) {
		track.established = true;
		track.visible = true;
	}
}

void Tracker::removeObsoleteTracks(vector<reference_wrapper<Track>>& unmatchedTracks) {
	for (Track& track : unmatchedTracks)
		track.visible = track.established && isVisible(track);
	for (auto track = tracks.begin(); track != tracks.end();) {
		if (track->visible)
			++track;
		else
			track = tracks.erase(track);
	}
}

bool Tracker::isVisible(const Track& track) const {
	return getCommonScore(track.state) > commonVisibilityThreshold
			|| (adaptive && getTargetScore(track) > targetVisibilityThreshold);
}

void Tracker::addNewTracks(const vector<Rect>& unmatchedDetections) {
	for (Rect detection : unmatchedDetections)
		tracks.push_back(createTrack(detection));
}

Track Tracker::createTrack(Rect target) {
	shared_ptr<LibSvmClassifier> svm = LibSvmClassifier::createBinarySvm(make_shared<LinearKernel>(), targetSvmC, true, false);
	svm->setPositiveExampleManagement(make_unique<AgeBasedExampleManagement>(positiveExampleCount));
	svm->setNegativeExampleManagement(make_unique<AgeBasedExampleManagement>(negativeExampleCount));
	unique_ptr<TrainableProbabilisticSvmClassifier> trainableProbabilisticSvm
			= make_unique<FixedTrainableProbabilisticSvmClassifier>(svm, 0.95, 0.05, 1.0, -1.0);
	shared_ptr<MeasurementModel> targetMeasurementModel = make_shared<ClassifierMeasurementModel>(
			featureExtractor, trainableProbabilisticSvm->getProbabilisticSvm());
	shared_ptr<RepellingMeasurementModel> repellingMeasurementModel = make_shared<RepellingMeasurementModel>();
	shared_ptr<MeasurementModel> measurementModel = make_shared<CombinedMeasurementModel>(
			commonMeasurementModel, targetMeasurementModel, repellingMeasurementModel);
	unique_ptr<ParticleFilter> filter = make_unique<ParticleFilter>(motionModel, measurementModel, particleCount);
	filter->initialize(versionedImage, target);
	return {
		nextTrackId++,
		std::move(trainableProbabilisticSvm),
		repellingMeasurementModel,
		std::move(filter),
		TargetState(target),
		false,
		false,
		false
	};
}

void Tracker::updateTargetModels() {
	for (Track& track : tracks) {
		if (track.visible && isAdaptingReasonable(track)) {
			adapt(track);
			track.adapted = true;
		} else {
			track.adapted = false;
		}
	}
}

bool Tracker::isAdaptingReasonable(const Track& track) const {
	return getCommonScore(track.state) > commonAdaptationThreshold
			|| getTargetScore(track) > targetAdaptationThreshold;
}

double Tracker::getCommonScore(const TargetState& state) const {
	shared_ptr<Patch> scorePatch = scoreExtractor->extract(state.x, state.y, state.width(), state.height());
	return scorePatch ? scorePatch->getData().at<float>(0, 0) : -1000;
}

double Tracker::getTargetScore(const Track& track) const {
	shared_ptr<Patch> featurePatch = featureExtractor->extract(track.state.x, track.state.y, track.state.width(), track.state.height());
	const classification::SvmClassifier& svm = *track.trainableProbabilisticSvm->getProbabilisticSvm()->getSvm();
	return featurePatch ? svm.computeHyperplaneDistance(featurePatch->getData()) : -1000;
}

void Tracker::adapt(Track& track) {
	Rect targetBounds = track.state.bounds();
	shared_ptr<SvmClassifier> svm = track.trainableProbabilisticSvm->getProbabilisticSvm()->getSvm();
	track.trainableProbabilisticSvm->retrain(
			getPositiveTrainingExamples(targetBounds),
			getNegativeTrainingExamples(targetBounds, *svm));
}

vector<Mat> Tracker::getPositiveTrainingExamples(Rect target) const {
	return vector<Mat>{ featureExtractor->extract(target)->getData() };
}

vector<Mat> Tracker::getNegativeTrainingExamples(Rect target, SvmClassifier& svm) const {
	vector<Mat> trainingExamples;
	int lowerX = target.x - target.width;
	int upperX = target.x + target.width;
	int lowerY = target.y - target.height;
	int upperY = target.y + target.height;
	int lowerH = target.height / 2;
	int upperH = target.height * 2;
	for (int i = 0; i < 20; ++i) {
		int x = std::uniform_int_distribution<int>{lowerX, upperX}(generator);
		int y = std::uniform_int_distribution<int>{lowerY, upperY}(generator);
		int height = std::uniform_int_distribution<int>{lowerH, upperH}(generator);
		int width = height * target.width / target.height;
		Rect candidate(x, y, width, height);
		if (computeOverlap(target, candidate) <= negativeOverlapThreshold) {
			shared_ptr<Patch> patch = featureExtractor->extract(candidate);
			if (patch && svm.computeHyperplaneDistance(patch->getData()) > -1)
				trainingExamples.push_back(patch->getData());
		}
	}
	return trainingExamples;
}

double Tracker::computeOverlap(Rect a, Rect b) const {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}

void Tracker::updateRepellingModels() {
	for (int i = 0; i < tracks.size(); ++i) {
		vector<TargetState> otherTargets;
		for (int k = 0; k < tracks.size(); ++k)
			if (k != i)
				otherTargets.push_back(tracks[k].state);
		tracks[i].repellingMeasurementModel->setOtherTargets(otherTargets);
	}
}

vector<pair<int, Rect>> Tracker::extractTargets() const {
	vector<pair<int, Rect>> idsAndBounds;
	for (const Track& track : tracks)
		if (track.visible)
			idsAndBounds.emplace_back(track.id, track.state.bounds());
	return idsAndBounds;
}

} // namespace tracking
