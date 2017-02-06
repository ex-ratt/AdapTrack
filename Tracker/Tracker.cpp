/*
 * Tracker.cpp
 *
 *  Created on: 26.10.2016
 *      Author: poschmann
 */

#include "stacktrace.hpp"
#include "StopWatch.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include "detection/AggregatedFeaturesDetector.hpp"
#include "detection/NonMaximumSuppression.hpp"
#include "imageio/CameraImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/DlibImageSource.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/extraction/ExactFhogExtractor.hpp"
#include "imageprocessing/filtering/FhogFilter.hpp"
#include "tracking/Tracker.hpp"
#include "tracking/filtering/RandomWalkModel.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using boost::optional;
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

struct Params {
	string svmFrontal;
	float thresholdFrontal;
	string svmAll;
	float thresholdAll;
	int binCount;
	int cellSize;
	int minSize;
	int maxSize;
};

struct TrackingEvaluation {
	double fppi;
	double mr;
	double overlapAverage;
	double iterationFps;
};

shared_ptr<ProbabilisticSvmClassifier> loadSvm(const string& filename, float threshold = 0);
shared_ptr<FhogFilter> createFhogFilter(int binCount, int cellSize);
shared_ptr<AggregatedFeaturesDetector> createDetector(
		shared_ptr<FhogFilter> fhogFilter, shared_ptr<SvmClassifier> svm, int cellSize, int minSize, int maxSize);
void run(Tracker& tracker, ImageSource& images);
void drawParticles(Mat& output, vector<pair<Rect, double>> particles);
void evaluate(Tracker& tracker, LabeledImageSource& images, int count);
TrackingEvaluation evaluate(Tracker& tracker, LabeledImageSource& images);
double computeOverlap(Rect a, Rect b);

int main(int argc, char **argv) {
	// TODO use program arguments or config files
//	shared_ptr<ImageSource> images = make_shared<CameraImageSource>(0);

//	shared_ptr<ImageSource> images = make_shared<DirectoryImageSource>("/home/ex-ratt/Videos/filesAb3/");
//	shared_ptr<DlibImageSource> images = make_shared<DlibImageSource>("/home/ex-ratt/Videos/filesAb3/annotations.xml");
//	Params params{"svm-fhog4-5x8", 1.5f, "svm-fp-fhog4-5x8", 0.0f, 4, 5, 50, 480}; // 2.3f

//	shared_ptr<ImageSource> images = make_shared<DirectoryImageSource>("/home/ex-ratt/Videos/a318/A4/");
//	Params params{"svm-fhog4-5x8", 1.5f, "svm-fp-fhog4-5x8", 0.0f, 4, 5, 50, 480};

//	shared_ptr<ImageSource> images = make_shared<DirectoryImageSource>("/home/ex-ratt/Videos/a318/group/");
	shared_ptr<DlibImageSource> images = make_shared<DlibImageSource>("/home/ex-ratt/Videos/a318/group/annotations.xml");
	Params params{"svm-fhog4-5x8", 2.2f, "svm-fp-fhog4-5x8", 0.0f, 4, 5, 50, 480};
//	Params params{"svm-fhog4-4x10", 0.8f, "svm-fp-fhog4-4x10", 0.0f, 4, 4, 50, 480};

//	shared_ptr<ImageSource> images = make_shared<DirectoryImageSource>("/home/ex-ratt/Documents/thesis-stuff/log-tdot-03-2013/2013_03_23_10-40-31/uEyeOmni_Image/OmniImage_2013-03-23_10-40-31/");
//	Params params{"svm-fhog4-5x8", 1.8f, "svm-fp-fhog4-5x8", 0.0f, 4, 3, 24, 240};

//	shared_ptr<ImageSource> images = make_shared<DirectoryImageSource>("/home/ex-ratt/Documents/thesis-stuff/log/2012_04_02_12-49-11/uEyeOmni_Image/OmniImage_2012-04-02_12-49-11/");
//	Params params{"svm-fhog4-5x8", 2.5f, "svm-fp-fhog4-5x8", 0.0f, 4, 5, 40, 240};

//	shared_ptr<ImageSource> images = make_shared<DirectoryImageSource>("/home/ex-ratt/Documents/thesis-stuff/log/2012_04_02_15-14-24/uEyeOmni_Image/OmniImage_2012-04-02_15-14-24/");
//	Params params{"svm-fhog4-5x8", 2.0f, "svm-fp-fhog4-5x8", 0.0f, 4, 4, 32, 240};

//	shared_ptr<ImageSource> images = make_shared<DirectoryImageSource>("/home/ex-ratt/Documents/thesis-stuff/log-2011-lndw/2011_7_1_18-20-55/uEyeOmni_Image/OmniImage_2011-07-01_18-20-55/");
//	Params params{"svm-fhog4-5x8", 2.5f, "svm-fp-fhog4-5x8", 0.0f, 4, 4, 32, 240};

	shared_ptr<ProbabilisticSvmClassifier> svm = loadSvm(params.svmFrontal, params.thresholdFrontal);
	shared_ptr<ProbabilisticSvmClassifier> svmAllHeads = loadSvm(params.svmAll, params.thresholdAll);
	svm->setLogisticA(0.0);
	svmAllHeads->setLogisticA(0.0);

	int windowWidth = svm->getSvm()->getSupportVectors()[0].cols;
	int windowHeight = svm->getSvm()->getSupportVectors()[0].rows;
	shared_ptr<FhogFilter> fhogFilter = createFhogFilter(params.binCount, params.cellSize);
	shared_ptr<ExactFhogExtractor> exactFhogExtractor = make_shared<ExactFhogExtractor>(fhogFilter, windowWidth, windowHeight);
	shared_ptr<AggregatedFeaturesDetector> detector = createDetector(fhogFilter, svm->getSvm(), params.cellSize, params.minSize, params.maxSize);
	shared_ptr<MotionModel> motionModel = make_shared<RandomWalkModel>(0.2, 0.05);
	unique_ptr<Tracker> headTracker = make_unique<Tracker>(exactFhogExtractor, detector, svmAllHeads, motionModel);
	headTracker->particleCount = 500;
	headTracker->adaptive = true;
	headTracker->associationThreshold = 0.3;
	headTracker->visibilityThreshold = -0.3;
	headTracker->negativeExampleCount = 10;
	headTracker->negativeOverlapThreshold = 0.5;
	headTracker->targetSvmC = 10;
	headTracker->learnRate = 0.5;
	run(*headTracker, *images);
//	cout << fixed << setprecision(2);
//	cout << "group " << params.svmFrontal
//			<< " threshold=" << params.thresholdFrontal
//			<< " particles=" << headTracker->particleCount
//			<< " C=" << headTracker->targetSvmC
//			<< " negativeOverlap=" << headTracker->negativeOverlapThreshold
//			<< " (hard)negatives=" << headTracker->negativeExampleCount
//			<< " learnRate=" << headTracker->learnRate
//			<< " visibility=" << headTracker->visibilityThreshold
//			<< endl;
//	evaluate(*headTracker, *images, 15);

	return EXIT_SUCCESS;
}

shared_ptr<ProbabilisticSvmClassifier> loadSvm(const string& filename, float threshold) {
	ifstream stream(filename);
	shared_ptr<ProbabilisticSvmClassifier> svm = ProbabilisticSvmClassifier::load(stream);
	stream.close();
	svm->getSvm()->setThreshold(threshold);
	return svm;
}

shared_ptr<FhogFilter> createFhogFilter(int binCount, int cellSize) {
	return make_shared<FhogFilter>(cellSize, binCount, false, true, 0.2f);
}

shared_ptr<AggregatedFeaturesDetector> createDetector(
		shared_ptr<FhogFilter> fhogFilter, shared_ptr<SvmClassifier> svm, int cellSize, int minSize, int maxSize) {
	int windowWidth = svm->getSupportVectors()[0].cols;
	int windowHeight = svm->getSupportVectors()[0].rows;
	shared_ptr<NonMaximumSuppression> nms = make_shared<NonMaximumSuppression>(
			0.3, NonMaximumSuppression::MaximumType::WEIGHTED_AVERAGE);
	return make_shared<AggregatedFeaturesDetector>(
			fhogFilter, cellSize, Size(windowWidth, windowHeight), 5, svm, nms, 1.0, 1.0, minSize, maxSize);
}

void run(Tracker& tracker, ImageSource& images) {
	GrayscaleFilter grayscaleFilter;
	Scalar colorUnconfirmed(0, 0, 0);
	vector<Scalar> colors = {
			Scalar(234, 82, 1), // blue
			Scalar(39, 187, 141), // green-yellow
			Scalar(1, 193, 252), // yellow-orange
			Scalar(35, 36, 228), // red
			Scalar(222, 24, 150), // violet
			Scalar(187, 150, 6), // blue-green
			Scalar(2, 237, 255), // yellow
			Scalar(0, 136, 255), // orange
			Scalar(141, 0, 223), // red-violet
			Scalar(189, 29, 71), // blue-violet
			Scalar(92, 142, 0), // green
			Scalar(0, 84, 255), // red-orange
	};
	int thickness = 2;
	Mat output;

	int frameCount = 0;
	duration<double> iterationTimeSum(0);
	bool run = true;
	bool pause = false;
	bool debug = false;
	while (run && images.next()) {
		++frameCount;
		Mat image = images.getImage();
		StopWatch iterationTimer = StopWatch::start();
		vector<pair<int, cv::Rect>> targets = tracker.update(grayscaleFilter.applyTo(image));
		milliseconds iterationTime = iterationTimer.stop();
		image.copyTo(output);
		if (debug) {
			for (const Track& track : tracker.getTracks()) {
				Mat intermediate = output.clone();
				if (!track.confirmed)
					rectangle(intermediate, track.state.bounds(), colorUnconfirmed, thickness);
				output = 0.75 * output + 0.25 * intermediate;
			}
			for (const Track& track : tracker.getTracks())
				drawParticles(output, track.particles());
		}
		for (const pair<int, cv::Rect>& target : targets)
			rectangle(output, target.second, colors[target.first % colors.size()], thickness);
		imshow("Detections", output);
		iterationTimeSum += iterationTime;
		float iterationFps = static_cast<double>(frameCount) / iterationTimeSum.count();
		cout << fixed << setprecision(1);
		cout << iterationTime.count() << " ms -> ";
		cout << iterationFps << " fps" << endl;
		char c = (char)waitKey(pause ? 0 : 2);
		if (c == 'q')
			run = false;
		else if (c == 'p')
			pause = !pause;
		else if (c == 'r')
			tracker.reset();
		else if (c == 'd')
			debug = !debug;
	}
}

void drawParticles(Mat& output, vector<pair<Rect, double>> particles) {
	if (particles.empty())
		return;
	Mat intermediate = output.clone();
	sort(particles.begin(), particles.end(), [](const auto& a, const auto& b) {
		return a.second < b.second;
	});
	double maxWeight = particles.back().second;
	double scale = 1.0 / maxWeight;
	for (const auto& particle : particles) {
		int x = particle.first.x + particle.first.width / 2;
		int y = particle.first.y + particle.first.height / 2;
		double weight = scale * particle.second;
		circle(intermediate, Point(x, y), 3, Scalar(weight * 255, weight * 255, weight * 255), 1);
	}
	output = 0.5 * output + 0.5 * intermediate;
}

void evaluate(Tracker& tracker, LabeledImageSource& images, int count) {
	TrackingEvaluation result;
	vector<TrackingEvaluation> results(count);
	for (int i = 0; i < count; ++i) {
		images.reset();
		tracker.reset();
		results[i] = evaluate(tracker, images);
		result.fppi += results[i].fppi;
		result.mr += results[i].mr;
		result.overlapAverage += results[i].overlapAverage;
		result.iterationFps += results[i].iterationFps;
	}
	result.fppi /= count;
	result.mr /= count;
	result.overlapAverage /= count;
	result.iterationFps /= count;

	cout << fixed << setprecision(3);
	cout << "Miss rate: " << result.mr << " (";
	for (int i = 0; i < count; ++i) {
		if (i > 0)
			cout << ", ";
		cout << results[i].mr;
	}
	cout << ")" << endl;

	cout << "FPPI: " << result.fppi << " (";
	for (int i = 0; i < count; ++i) {
		if (i > 0)
			cout << ", ";
		cout << results[i].fppi;
	}
	cout << ")" << endl;

	cout << "Avg Overlap: " << result.overlapAverage << " (";
	for (int i = 0; i < count; ++i) {
		if (i > 0)
			cout << ", ";
		cout << results[i].overlapAverage;
	}
	cout << ")" << endl;

	cout << std::fixed << std::setprecision(1);
	cout << "FPS: " << result.iterationFps << " (";
	for (int i = 0; i < count; ++i) {
		if (i > 0)
			cout << ", ";
		cout << results[i].iterationFps;
	}
	cout << ")" << endl;
}

TrackingEvaluation evaluate(Tracker& tracker, LabeledImageSource& images) {
	GrayscaleFilter grayscaleFilter;
	int frames = 0;
	int truePositives = 0;
	int falsePositives = 0;
	int falseNegatives = 0;
	double overlapSum = 0;
	duration<double> iterationTimeSum(0);
	while (images.next()) {
		++frames;
		Mat image = images.getImage();
		LandmarkCollection collection = images.getLandmarks();
		StopWatch iterationTimer = StopWatch::start();
		vector<pair<int, Rect>> targets = tracker.update(grayscaleFilter.applyTo(image));
		milliseconds iterationTime = iterationTimer.stop();
		iterationTimeSum += iterationTime;
		for (const shared_ptr<Landmark> landmark : collection.getLandmarks()) {
			if (landmark->isVisible()) {
				bool ignore = landmark->getName().compare(0, 6, "ignore") == 0;
				if (!ignore)
					++falseNegatives;
				for (auto target = targets.begin(); target != targets.end(); ++target) {
					if (computeOverlap(landmark->getRect(), target->second) >= 0.5) {
						if (!ignore) {
							--falseNegatives;
							++truePositives;
						}
						overlapSum += computeOverlap(landmark->getRect(), target->second);
						targets.erase(target);
						break;
					}
				}
			}
		}
		falsePositives += targets.size();
	}
	TrackingEvaluation result;
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
