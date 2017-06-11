/*
 * MultiTrackerEvaluator.cpp
 *
 *  Created on: 26.10.2016
 *      Author: poschmann
 */

#include "stacktrace.hpp"
#include "StopWatch.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "detection/AggregatedFeaturesDetector.hpp"
#include "detection/NonMaximumSuppression.hpp"
#include "imageio/DlibImageSource.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/extraction/ExactFhogExtractor.hpp"
#include "imageprocessing/filtering/FhogFilter.hpp"
#include "tracking/MultiTracker.hpp"
#include "tracking/filtering/RandomWalkModel.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace classification;
using namespace cv;
using namespace detection;
using namespace imageio;
using namespace imageprocessing;
using namespace imageprocessing::extraction;
using namespace imageprocessing::filtering;
using namespace std;
using namespace std::chrono;
using namespace tracking;
using namespace tracking::filtering;

struct TrackingEvaluation {
	double fppi;
	double mr;
	double overlapAverage;
	double iterationFps;
};

shared_ptr<ProbabilisticSvmClassifier> loadSvm(const string& filename, float threshold = 0);
shared_ptr<FhogFilter> createFhogFilter(int binCount, int cellSize);
shared_ptr<AggregatedFeaturesDetector> createDetector(
		shared_ptr<FhogFilter> fhogFilter, shared_ptr<SvmClassifier> svm, int cellSize, int minWidth, int maxWidth);
void evaluate(MultiTracker& tracker, AnnotatedImageSource& images, double aspectRatio, int count);
TrackingEvaluation evaluate(MultiTracker& tracker, const vector<AnnotatedImage>& images);
double computeOverlap(Rect a, Rect b);

int main(int argc, char **argv) {
	if (argc != 6) {
		cout << "usage: " << argv[0] << " annotation svm cellsize detectionThreshold visibilityThreshold" << endl;
		cout << "  annotation: XML-file that contains image paths and annotations in dlib format" << endl;
		cout << "  svm: text file that contains SVM data (e.g. created by DetectorTrainer)" << endl;
		cout << "  cellsize: size of the square FHOG cells in pixels" << endl;
		cout << "  detectionThreshold: SVM score threshold for detections to be reported" << endl;
		cout << "  visibilityThreshold: SVM score threshold for tracks to be regarded visible" << endl;
		return EXIT_FAILURE;
	}
	string annotationFile = argv[1];
	string svmFile = argv[2];
	int cellSize = std::stoi(argv[3]);
	float detectionThreshold = std::stof(argv[4]);
	float visibilityThreshold = std::stof(argv[5]);
	int minWidth = 0;
	int maxWidth = 0;
	int repetitions = 25;

	shared_ptr<DlibImageSource> images = make_shared<DlibImageSource>(annotationFile);
	shared_ptr<ProbabilisticSvmClassifier> svm = loadSvm(svmFile, detectionThreshold);
	int binCount = (svm->getSvm()->getSupportVectors()[0].channels() - 4) / 3;
	int windowWidth = svm->getSvm()->getSupportVectors()[0].cols;
	int windowHeight = svm->getSvm()->getSupportVectors()[0].rows;

	shared_ptr<FhogFilter> fhogFilter = createFhogFilter(binCount, cellSize);
	shared_ptr<ExactFhogExtractor> exactFhogExtractor = make_shared<ExactFhogExtractor>(fhogFilter, windowWidth, windowHeight);
	shared_ptr<AggregatedFeaturesDetector> detector = createDetector(fhogFilter, svm->getSvm(), cellSize, minWidth, maxWidth);
	shared_ptr<MotionModel> motionModel = make_shared<RandomWalkModel>(0.2, 0.05);
	unique_ptr<MultiTracker> tracker = make_unique<MultiTracker>(exactFhogExtractor, detector, svm, motionModel);
	tracker->particleCount = 500;
	tracker->adaptive = true;
	tracker->associationThreshold = 0.3;
	tracker->visibilityThreshold = visibilityThreshold;
	tracker->negativeExampleCount = 10;
	tracker->negativeOverlapThreshold = 0.5;
	tracker->targetSvmC = 10;
	tracker->learnRate = 0.5;
	cout << fixed << setprecision(2);
	cout << annotationFile
			<< (tracker->adaptive ? " adaptive" : " non-adaptive")
			<< " particles=" << tracker->particleCount
			<< " " << svmFile
			<< " threshold=" << detectionThreshold
			<< " visibility=" << tracker->visibilityThreshold;
	if (tracker->adaptive)
	cout << " negativeOverlap=" << tracker->negativeOverlapThreshold
			<< " (hard)negatives=" << tracker->negativeExampleCount
			<< " C=" << tracker->targetSvmC
			<< " learnRate=" << tracker->learnRate;
	cout << endl;
	evaluate(*tracker, *images, static_cast<double>(windowWidth) / windowHeight, repetitions);

	return EXIT_SUCCESS;
}

shared_ptr<ProbabilisticSvmClassifier> loadSvm(const string& filename, float threshold) {
	ifstream stream(filename);
	shared_ptr<ProbabilisticSvmClassifier> svm = ProbabilisticSvmClassifier::load(stream);
	stream.close();
	svm->getSvm()->setThreshold(threshold);
	svm->setLogisticA(0.0);
	return svm;
}

shared_ptr<FhogFilter> createFhogFilter(int binCount, int cellSize) {
	return make_shared<FhogFilter>(cellSize, binCount, false, true, 0.2f);
}

shared_ptr<AggregatedFeaturesDetector> createDetector(
		shared_ptr<FhogFilter> fhogFilter, shared_ptr<SvmClassifier> svm, int cellSize, int minWidth, int maxWidth) {
	int windowWidth = svm->getSupportVectors()[0].cols;
	int windowHeight = svm->getSupportVectors()[0].rows;
	shared_ptr<NonMaximumSuppression> nms = make_shared<NonMaximumSuppression>(
			0.3, NonMaximumSuppression::MaximumType::MAX_SCORE);
	return make_shared<AggregatedFeaturesDetector>(
			fhogFilter, cellSize, Size(windowWidth, windowHeight), 5, svm, nms, 1.0, 1.0, minWidth, maxWidth);
}

void evaluate(MultiTracker& tracker, AnnotatedImageSource& source, double aspectRatio, int count) {
	vector<AnnotatedImage> images;
	while (source.next())
		images.push_back(source.getAnnotatedImage());
	for (AnnotatedImage& image : images)
		image.annotations.adjustSizes(aspectRatio);

	TrackingEvaluation mean{0,0,0,0};
	vector<TrackingEvaluation> results(count);
	for (int i = 0; i < count; ++i) {
		tracker.reset();
		results[i] = evaluate(tracker, images);
		mean.fppi += results[i].fppi;
		mean.mr += results[i].mr;
		mean.overlapAverage += results[i].overlapAverage;
		mean.iterationFps += results[i].iterationFps;
	}
	mean.fppi /= count;
	mean.mr /= count;
	mean.overlapAverage /= count;
	mean.iterationFps /= count;

	TrackingEvaluation variance{0,0,0,0};
	for (int i = 0; i < count; ++i) {
		variance.fppi += (mean.fppi - results[i].fppi) * (mean.fppi - results[i].fppi);
		variance.mr += (mean.mr - results[i].mr) * (mean.mr - results[i].mr);
		variance.overlapAverage += (mean.overlapAverage - results[i].overlapAverage) * (mean.overlapAverage - results[i].overlapAverage);
		variance.iterationFps += (mean.iterationFps - results[i].iterationFps) * (mean.iterationFps - results[i].iterationFps);
	}
	variance.fppi /= count;
	variance.mr /= count;
	variance.overlapAverage /= count;
	variance.iterationFps /= count;

	cout << fixed << setprecision(4);
	cout << "Miss rate: " << mean.mr << "/" << sqrt(variance.mr) << " (";
	for (int i = 0; i < count; ++i) {
		if (i > 0)
			cout << ", ";
		cout << results[i].mr;
	}
	cout << ")" << endl;

	cout << fixed << setprecision(5);
	cout << "FPPI: " << mean.fppi << "/" << sqrt(variance.fppi) << " (";
	for (int i = 0; i < count; ++i) {
		if (i > 0)
			cout << ", ";
		cout << results[i].fppi;
	}
	cout << ")" << endl;

	cout << fixed << setprecision(3);
	cout << "Avg Overlap: " << mean.overlapAverage << "/" << sqrt(variance.overlapAverage) << " (";
	for (int i = 0; i < count; ++i) {
		if (i > 0)
			cout << ", ";
		cout << results[i].overlapAverage;
	}
	cout << ")" << endl;

	cout << std::fixed << std::setprecision(1);
	cout << "FPS: " << mean.iterationFps << "/" << sqrt(variance.iterationFps) << " (";
	for (int i = 0; i < count; ++i) {
		if (i > 0)
			cout << ", ";
		cout << results[i].iterationFps;
	}
	cout << ")" << endl;
}

TrackingEvaluation evaluate(MultiTracker& tracker, const vector<AnnotatedImage>& images) {
	GrayscaleFilter grayscaleFilter;
	int frames = 0;
	int truePositives = 0;
	int falsePositives = 0;
	int falseNegatives = 0;
	double overlapSum = 0;
	duration<double> iterationTimeSum(0);
	for (const AnnotatedImage& image : images) {
		++frames;
		Mat frame = image.image;
		StopWatch iterationTimer = StopWatch::start();
		vector<pair<int, Rect>> targets = tracker.update(grayscaleFilter.applyTo(frame));
		milliseconds iterationTime = iterationTimer.stop();
		iterationTimeSum += iterationTime;
		for (Rect annotation : image.annotations.positiveAnnotations()) {
			++falseNegatives;
			for (auto target = targets.begin(); target != targets.end(); ++target) {
				if (computeOverlap(annotation, target->second) >= 0.5) {
					--falseNegatives;
					++truePositives;
					overlapSum += computeOverlap(annotation, target->second);
					targets.erase(target);
					break;
				}
			}
		}
		for (Rect annotation : image.annotations.fuzzyAnnotations()) {
			for (auto target = targets.begin(); target != targets.end(); ++target) {
				if (computeOverlap(annotation, target->second) >= 0.5) {
					targets.erase(target);
					break;
				}
			}
		}
		falsePositives += targets.size();
	}
	TrackingEvaluation result{0,0,0,0};
	result.fppi = falsePositives / static_cast<double>(frames);
	result.mr = falseNegatives / static_cast<double>(falseNegatives + truePositives);
	result.overlapAverage = overlapSum / truePositives;
	result.iterationFps = static_cast<double>(frames) / iterationTimeSum.count();
	return result;
}

double computeOverlap(Rect a, Rect b) {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}
