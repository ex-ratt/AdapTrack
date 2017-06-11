/*
 * Tracker.cpp
 *
 *  Created on: 02.11.2016
 *      Author: poschmann
 */

#include "classification/IncrementalLinearSvmTrainer.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/PseudoProbabilisticSvmTrainer.hpp"
#include "libsvm/LibSvmTrainer.hpp"
#include "tracking/MultiTracker.hpp"
#include "tracking/filtering/ClassifierMeasurementModel.hpp"
#include "tracking/filtering/CorrelatedCombinationModel.hpp"
#include "imageprocessing/Patch.hpp"

using classification::IncrementalLinearSvmTrainer;
using classification::LinearKernel;
using classification::ProbabilisticSvmClassifier;
using classification::PseudoProbabilisticSvmTrainer;
using classification::SvmClassifier;
using cv::Mat;
using cv::Point;
using cv::Rect;
using detection::AggregatedFeaturesDetector;
using libsvm::LibSvmTrainer;
using tracking::filtering::ClassifierMeasurementModel;
using tracking::filtering::CorrelatedCombinationModel;
using tracking::filtering::MeasurementModel;
using tracking::filtering::MotionModel;
using tracking::filtering::ParticleFilter;
using tracking::filtering::TargetState;
using imageprocessing::Patch;
using imageprocessing::VersionedImage;
using imageprocessing::extraction::FeatureExtractor;
using std::make_shared;
using std::make_unique;
using std::pair;
using std::reference_wrapper;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace tracking {

MultiTracker::MultiTracker(shared_ptr<FeatureExtractor> exactFeatureExtractor,
		shared_ptr<AggregatedFeaturesDetector> detector,
		shared_ptr<ProbabilisticSvmClassifier> svm,
		shared_ptr<MotionModel> motionModel) :
				generator(std::random_device()()),
				versionedImage(make_shared<VersionedImage>()),
				tracks(),
				nextTrackId(0),
				detector(detector),
				pyramidFeatureExtractor(detector->getFeatureExtractor()),
				exactFeatureExtractor(exactFeatureExtractor),
				svm(svm),
				commonMeasurementModel(make_shared<ClassifierMeasurementModel>(pyramidFeatureExtractor, svm)),
				motionModel(motionModel),
				particleCount(500),
				adaptive(true),
				associationThreshold(0.333),
				visibilityThreshold(0.0),
				negativeExampleCount(10),
				negativeOverlapThreshold(0.5),
				targetSvmC(10),
				learnRate(0.5) {}

const vector<Track>& MultiTracker::getTracks() const {
	return tracks;
}

void MultiTracker::reset() {
	tracks.clear();
}

vector<pair<int, Rect>> MultiTracker::update(const Mat& image) {
	updateImage(image);
	updateFilters();
	vector<Rect> detections = detector->detect(versionedImage);
	Associations associations = pickAssociations(tracks, detections);
	confirmMatchedTracks(associations.matchedTracks);
	removeObsoleteTracks(associations.unmatchedTracks);
	removeOverlappingTracks();
	addNewTracks(associations.unmatchedDetections);
	if (adaptive)
		updateTargetModels();
	return extractTargets();
}

void MultiTracker::updateImage(const Mat& image) {
	versionedImage->setData(image);
	pyramidFeatureExtractor->update(versionedImage);
	exactFeatureExtractor->update(versionedImage);
}

void MultiTracker::updateFilters() {
	for (Track& track : tracks) {
		track.state = track.filter->update(versionedImage);
		shared_ptr<Patch> patch = exactFeatureExtractor->extract(
				track.state.x, track.state.y, track.state.width(), track.state.height());
		if (patch) {
			track.features = patch->getData();
			classification::SvmClassifier& svm = adaptive ? *track.svm->getSvm() : *this->svm->getSvm();
			track.score = svm.computeHyperplaneDistance(track.features);
		} else {
			track.features = Mat();
			track.score = -100.0;
		}
	}
}

Associations MultiTracker::pickAssociations(vector<Track>& tracks, vector<Rect>& detections) const {
	Associations associations;
	Mat overlaps(tracks.size(), detections.size(), CV_32FC1);
	for (int i = 0; i < tracks.size(); ++i) {
		for (int j = 0; j < detections.size(); ++j) {
			overlaps.at<float>(i, j) = computeOverlap(tracks[i].state.bounds(), detections[j]);
		}
	}
	vector<int> unmatchedTrackIndices(tracks.size());
	vector<int> unmatchedDetectionIndices(detections.size());
	std::iota(unmatchedTrackIndices.begin(), unmatchedTrackIndices.end(), 0);
	std::iota(unmatchedDetectionIndices.begin(), unmatchedDetectionIndices.end(), 0);
	cv::Point nextMatch = getBestMatch(overlaps, associationThreshold, unmatchedTrackIndices, unmatchedDetectionIndices);
	while (nextMatch.x >= 0) {
		associations.matchedTracks.push_back(std::ref(tracks[nextMatch.y]));
		unmatchedTrackIndices.erase(std::find(unmatchedTrackIndices.begin(), unmatchedTrackIndices.end(), nextMatch.y));
		unmatchedDetectionIndices.erase(std::find(unmatchedDetectionIndices.begin(), unmatchedDetectionIndices.end(), nextMatch.x));
		nextMatch = getBestMatch(overlaps, associationThreshold, unmatchedTrackIndices, unmatchedDetectionIndices);
	}
	for (int i : unmatchedTrackIndices)
		associations.unmatchedTracks.push_back(std::ref(tracks[i]));
	for (int j : unmatchedDetectionIndices)
		associations.unmatchedDetections.push_back(detections[j]);
	return associations;
}

Point MultiTracker::getBestMatch(const Mat& overlaps, float threshold,
		const vector<int>& unmatchedTrackIndices, const vector<int>& unmatchedDetectionIndices) const {
	Point maxElement(-1, -1);
	float maxOverlap = threshold;
	for (int trackIndex : unmatchedTrackIndices) {
		for (int detectionIndex : unmatchedDetectionIndices) {
			if (overlaps.at<float>(trackIndex, detectionIndex) > maxOverlap) {
				maxOverlap = overlaps.at<float>(trackIndex, detectionIndex);
				maxElement.y = trackIndex;
				maxElement.x = detectionIndex;
			}
		}
	}
	return maxElement;
}

void MultiTracker::confirmMatchedTracks(vector<reference_wrapper<Track>>& matchedTracks) {
	for (Track& track : matchedTracks) {
		if (!track.confirmed) {
			track.confirmed = true;
			track.id = nextTrackId++;
		}
	}
}

void MultiTracker::removeObsoleteTracks(vector<reference_wrapper<Track>>& unmatchedTracks) {
	for (const Track& unmatchedTrack : unmatchedTracks) {
		if (!unmatchedTrack.confirmed || !isVisible(unmatchedTrack)) {
			tracks.erase(std::find_if(tracks.begin(), tracks.end(), [&](const Track& track) {
				return &track == &unmatchedTrack;
			}));
		}
	}
}

bool MultiTracker::isVisible(const Track& track) const {
	bool isTargetInsideFeaturePyramid = !!pyramidFeatureExtractor->extract(track.state.bounds());
	return track.score > visibilityThreshold && isTargetInsideFeaturePyramid;
}

void MultiTracker::removeOverlappingTracks() {
	for (auto track1 = tracks.begin(); track1 != tracks.end();) {
		for (auto track2 = track1 + 1; track2 != tracks.end();) {
			if (computeOverlap(track1->state.bounds(), track2->state.bounds()) > associationThreshold) {
				if (track1->score > track2->score) {
					track2 = tracks.erase(track2);
				} else {
					track1 = tracks.erase(track1) - 1;
					break;
				}
			} else { // tracks do not overlap (by much)
				++track2;
			}
		}
		++track1;
	}
}

void MultiTracker::addNewTracks(const vector<Rect>& unmatchedDetections) {
	for (Rect detection : unmatchedDetections)
		tracks.push_back(createTrack(detection));
}

Track MultiTracker::createTrack(Rect target) {
	auto probabilisticSvm = make_shared<ProbabilisticSvmClassifier>(make_shared<LinearKernel>());
	auto libSvmTrainer = make_shared<LibSvmTrainer>(targetSvmC, true);
	auto incrementalSvmTrainer = make_shared<IncrementalLinearSvmTrainer>(libSvmTrainer, learnRate);
	auto probabilisticSvmTrainer = make_shared<PseudoProbabilisticSvmTrainer>(incrementalSvmTrainer, 0.95, 0.05, 1.0, -1.0);
	shared_ptr<MeasurementModel> targetMeasurementModel = make_shared<ClassifierMeasurementModel>(
			pyramidFeatureExtractor, probabilisticSvm);
	shared_ptr<MeasurementModel> measurementModel = adaptive
			? make_shared<CorrelatedCombinationModel>(commonMeasurementModel, targetMeasurementModel)
					: commonMeasurementModel;
	unique_ptr<ParticleFilter> filter = make_unique<ParticleFilter>(motionModel, measurementModel, particleCount);
	filter->initialize(versionedImage, target);
	return {
		0,
		probabilisticSvm,
		probabilisticSvmTrainer,
		std::move(filter),
		TargetState(target),
		false,
		Mat(),
		0.0
	};
}

void MultiTracker::updateTargetModels() {
	for (Track& track : tracks) {
		if (track.confirmed)
			adapt(track);
	}
}

void MultiTracker::adapt(Track& track) {
	Rect targetBounds = track.state.bounds();
	if (track.svm->getSvm()->getSupportVectors().empty())
		track.svmTrainer->train(*track.svm,
				vector<Mat>{track.features}, getNegativeTrainingExamples(targetBounds));
	else
		track.svmTrainer->retrain(*track.svm,
				vector<Mat>{track.features}, getNegativeTrainingExamples(targetBounds, *track.svm->getSvm()));
}

vector<Mat> MultiTracker::getNegativeTrainingExamples(Rect target) const {
	int lowerX = target.x - target.width;
	int upperX = target.x + target.width;
	int lowerY = target.y - target.height;
	int upperY = target.y + target.height;
	int lowerH = target.height / 2;
	int upperH = target.height * 2;
	vector<Mat> trainingExamples;
	trainingExamples.reserve(negativeExampleCount);
	while (trainingExamples.size() < trainingExamples.capacity()) {
		int x = std::uniform_int_distribution<int>{lowerX, upperX}(generator);
		int y = std::uniform_int_distribution<int>{lowerY, upperY}(generator);
		int height = std::uniform_int_distribution<int>{lowerH, upperH}(generator);
		int width = height * target.width / target.height;
		shared_ptr<Patch> patch = pyramidFeatureExtractor->extract(Rect(x, y, width, height));
		if (patch && computeOverlap(target, patch->getBounds()) <= negativeOverlapThreshold)
			trainingExamples.push_back(patch->getData());
	}
	return trainingExamples;
}

vector<Mat> MultiTracker::getNegativeTrainingExamples(Rect target, const SvmClassifier& svm) const {
	int lowerX = target.x - target.width;
	int upperX = target.x + target.width;
	int lowerY = target.y - target.height;
	int upperY = target.y + target.height;
	int lowerH = target.height / 2;
	int upperH = target.height * 2;
	vector<pair<double, Mat>> trainingCandidates;
	trainingCandidates.reserve(3 * negativeExampleCount);
	while (trainingCandidates.size() < trainingCandidates.capacity()) {
		int x = std::uniform_int_distribution<int>{lowerX, upperX}(generator);
		int y = std::uniform_int_distribution<int>{lowerY, upperY}(generator);
		int height = std::uniform_int_distribution<int>{lowerH, upperH}(generator);
		int width = height * target.width / target.height;
		shared_ptr<Patch> patch = pyramidFeatureExtractor->extract(Rect(x, y, width, height));
		if (patch && computeOverlap(target, patch->getBounds()) <= negativeOverlapThreshold) {
			double score = svm.computeHyperplaneDistance(patch->getData());
			trainingCandidates.emplace_back(score, patch->getData());
		}
	}
	std::partial_sort(trainingCandidates.begin(), trainingCandidates.begin() + negativeExampleCount, trainingCandidates.end(),
			[](const auto& a, const auto& b) { return a.first > b.first; });
	vector<Mat> trainingExamples;
	trainingExamples.reserve(negativeExampleCount);
	for (int i = 0; i < trainingExamples.capacity(); ++i)
		trainingExamples.push_back(trainingCandidates[i].second);
	return trainingExamples;
}

double MultiTracker::computeOverlap(Rect a, Rect b) const {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}

vector<pair<int, Rect>> MultiTracker::extractTargets() const {
	vector<pair<int, Rect>> idsAndBounds;
	for (const Track& track : tracks)
		if (track.confirmed)
			idsAndBounds.emplace_back(track.id, track.state.bounds());
	return idsAndBounds;
}

} // namespace tracking
