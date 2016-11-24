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
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/filtering/FhogFilter.hpp"
#include "tracking/Tracker.hpp"
#include "tracking/filtering/ConstantVelocityModel.hpp"
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

shared_ptr<ProbabilisticSvmClassifier> loadSvm(const string& filename, float threshold = 0);
shared_ptr<AggregatedFeaturesDetector> createDetector(shared_ptr<SvmClassifier> svm, int binCount, int cellSize, int windowSize);
void run(Tracker& tracker, ImageSource& images);
void drawParticles(Mat& output, vector<pair<Rect, double>> particles);

int main(int argc, char **argv) {
	// TODO use program arguments or config files
//	shared_ptr<ImageSource> images = make_shared<CameraImageSource>(0);
//	shared_ptr<ImageSource> images = make_shared<DirectoryImageSource>("/home/poschmann/Videos/coglog/filesAb3/");
//	shared_ptr<ImageSource> images = make_shared<DirectoryImageSource>("/home/poschmann/Videos/a318/A4/");
	shared_ptr<ImageSource> images = make_shared<DirectoryImageSource>("/home/poschmann/Videos/a318/group/");

	// FHOG9 4x10
//	shared_ptr<ProbabilisticSvmClassifier> svm = loadSvm("svm-fhog9-4x10", -0.0730626f);
//	shared_ptr<AggregatedFeaturesDetector> detector = createDetector(svm->getSvm(), 9, 4, 10);
	// FHOG9 5x8
//	shared_ptr<ProbabilisticSvmClassifier> svm = loadSvm("svm-fhog9-5x8", 0.289107f);
//	shared_ptr<AggregatedFeaturesDetector> detector = createDetector(svm->getSvm(), 9, 5, 8);
	// FHOG4 4x10
//	shared_ptr<ProbabilisticSvmClassifier> svm = loadSvm("svm-fhog4-4x10", 0.175828f);
//	shared_ptr<AggregatedFeaturesDetector> detector = createDetector(svm->getSvm(), 4, 4, 10);
	// FHOG4 4x9
//	shared_ptr<ProbabilisticSvmClassifier> svm = loadSvm("svm-fhog4-4x9", 0.258097f);
//	shared_ptr<AggregatedFeaturesDetector> detector = createDetector(svm->getSvm(), 4, 4, 9);
	// FHOG4 4x8
//	shared_ptr<ProbabilisticSvmClassifier> svm = loadSvm("svm-fhog4-4x8", 0.393527f);
//	shared_ptr<AggregatedFeaturesDetector> detector = createDetector(svm->getSvm(), 4, 4, 8);
	// FHOG4 5x8
	shared_ptr<ProbabilisticSvmClassifier> svm = loadSvm("svm-fhog4-5x8", 2.24504f); // 0.757608f
//	shared_ptr<ProbabilisticSvmClassifier> svm = loadSvm("svm-fhog4-5x8", 1.29315f); // 0.757608f
	shared_ptr<AggregatedFeaturesDetector> detector = createDetector(svm->getSvm(), 4, 5, 8);

	shared_ptr<MotionModel> motionModel = make_shared<ConstantVelocityModel>(0.02);
	unique_ptr<Tracker> headTracker = make_unique<Tracker>(detector, svm, motionModel);
	headTracker->particleCount = 500;
	headTracker->adaptive = true;
	headTracker->commonVisibilityThreshold = -1.0;
	headTracker->targetVisibilityThreshold = -0.5;
	headTracker->commonAdaptationThreshold = 0.0;
	headTracker->targetAdaptationThreshold = 0.0;
	headTracker->targetSvmC = 10;
	headTracker->positiveExampleCount = 1;
	headTracker->negativeExampleCount = 20;
	run(*headTracker, *images);

	return EXIT_SUCCESS;
}

shared_ptr<ProbabilisticSvmClassifier> loadSvm(const string& filename, float threshold) {
	std::ifstream stream(filename);
	shared_ptr<ProbabilisticSvmClassifier> svm = ProbabilisticSvmClassifier::load(stream);
	stream.close();
	svm->getSvm()->setThreshold(threshold);
	return svm;
}

shared_ptr<AggregatedFeaturesDetector> createDetector(shared_ptr<SvmClassifier> svm, int binCount, int cellSize, int windowSize) {
	int octaveLayerCount = 5;
	double overlapThreshold = 0.3;
	shared_ptr<FhogFilter> fhogFilter = make_shared<FhogFilter>(cellSize, binCount, false, true, 0.2f);
	// TODO WEIGHTED_AVERAGE is bad as scores can be negative or start at non-null
	shared_ptr<NonMaximumSuppression> nms = make_shared<NonMaximumSuppression>(overlapThreshold, NonMaximumSuppression::MaximumType::AVERAGE);
	return make_shared<AggregatedFeaturesDetector>(make_shared<GrayscaleFilter>(), fhogFilter,
			cellSize, cv::Size(windowSize, windowSize), octaveLayerCount, svm, nms, 1.0, 1.0, 50); // TODO min height 50 (for now)
}

void run(Tracker& tracker, ImageSource& images) {
	Mat output;
	Scalar colorInitializing(255, 102, 0);
	Scalar colorAdapted(255, 255, 255);
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

	int frameCount = 0;
	duration<double> iterationTimeSum(0);
	bool run = true;
	bool pause = false;
	bool debug = true;
	while (run && images.next()) {
		++frameCount;
		Mat image = images.getImage();
		StopWatch iterationTimer = StopWatch::start();
		vector<pair<int, cv::Rect>> targets = tracker.update(image);
		image.copyTo(output);
		if (debug) {
			for (const Track& track : tracker.getTracks()) {
				Mat intermediate = output.clone();
				if (!track.established)
					cv::rectangle(intermediate, track.state.bounds(), colorInitializing, CV_FILLED);
				else if (track.adapted)
					cv::rectangle(intermediate, track.state.bounds(), colorAdapted, CV_FILLED);
				output = 0.75 * output + 0.25 * intermediate;
			}
			for (const Track& track : tracker.getTracks())
				drawParticles(output, track.particles());
		}
		for (const pair<int, cv::Rect>& target : targets) {
			cv::rectangle(output, target.second, colors[target.first % colors.size()], thickness);
		}
		cv::imshow("Detections", output);
		milliseconds iterationTime = iterationTimer.stop();
		char c = (char)cv::waitKey(pause ? 0 : 5);
		if (c == 'q')
			run = false;
		else if (c == 'p')
			pause = !pause;
		else if (c == 'r')
			tracker.reset();
		else if (c == 'd')
			debug = !debug;
		iterationTimeSum += iterationTime;
		float iterationFps = static_cast<double>(frameCount) / iterationTimeSum.count();
		std::cout << std::fixed << std::setprecision(1);
		std::cout << iterationTime.count() << " ms -> ";
		std::cout << iterationFps << "fps" << std::endl;
	}
}

void drawParticles(Mat& output, vector<pair<Rect, double>> particles) {
	if (particles.empty())
		return;
	Mat intermediate = output.clone();
	std::sort(particles.begin(), particles.end(), [](const auto& a, const auto& b) {
		return a.second < b.second;
	});
	double maxWeight = particles.back().second;
	double scale = 1.0 / maxWeight;
	for (const auto& particle : particles) {
		int x = particle.first.x + particle.first.width / 2;
		int y = particle.first.y + particle.first.height / 2;
		double weight = scale * particle.second;
		cv::circle(intermediate, Point(x, y), 3, Scalar(weight * 255, weight * 255, weight * 255), 1);
	}
	output = 0.5 * output + 0.5 * intermediate;
}
