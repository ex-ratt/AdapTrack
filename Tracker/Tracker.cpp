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
shared_ptr<AggregatedFeaturesDetector> createDetector(shared_ptr<SvmClassifier> svm, int cellSize, int windowSize);
void run(Tracker& tracker, ImageSource& images);
void drawParticles(Mat& output, vector<pair<Rect, double>> particles);

int main(int argc, char **argv) {
	// TODO use program arguments or config files
//	shared_ptr<ImageSource> images = make_shared<CameraImageSource>(0);
	shared_ptr<ImageSource> images = make_shared<DirectoryImageSource>("/home/poschmann/Videos/coglog/filesAb3/");

//	shared_ptr<ProbabilisticSvmClassifier> svm = loadSvm("svm-fhog4x10", -0.0730626f);
//	shared_ptr<AggregatedFeaturesDetector> detector = createDetector(svm->getSvm(), 4, 10);
	shared_ptr<ProbabilisticSvmClassifier> svm = loadSvm("svm-fhog5x8", 0.289107f);
	shared_ptr<AggregatedFeaturesDetector> detector = createDetector(svm->getSvm(), 5, 8);
	shared_ptr<MotionModel> motionModel = make_shared<ConstantVelocityModel>(0.05);
	unique_ptr<Tracker> headTracker = make_unique<Tracker>(detector, svm, motionModel, 500);
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

shared_ptr<AggregatedFeaturesDetector> createDetector(shared_ptr<SvmClassifier> svm, int cellSize, int windowSize) {
	int octaveLayerCount = 5;
	double overlapThreshold = 0.3;
	shared_ptr<FhogFilter> fhogFilter = make_shared<FhogFilter>(cellSize, 9, false, true, 0.2f);
	shared_ptr<NonMaximumSuppression> nms = make_shared<NonMaximumSuppression>(overlapThreshold, NonMaximumSuppression::MaximumType::WEIGHTED_AVERAGE);
	return make_shared<AggregatedFeaturesDetector>(make_shared<GrayscaleFilter>(), fhogFilter,
			cellSize, cv::Size(windowSize, windowSize), octaveLayerCount, svm, nms, 1.0, 1.0, 60); // TODO min height 60 (for now)
}

void run(Tracker& tracker, ImageSource& images) {
	Mat output;
	cv::Scalar color(0, 255, 0);
	int thickness = 2;

	int frameCount = 0;
	duration<double> iterationTimeSum(0);
	bool run = true;
	bool pause = false;
	bool showParticles = true;
	while (run && images.next()) {
		++frameCount;
		StopWatch iterationTimer = StopWatch::start();
		vector<Rect> targets = tracker.update(images.getImage());
		images.getImage().copyTo(output);
		if (showParticles)
			drawParticles(output, tracker.getParticleLocations());
		for (const Rect& target : targets)
			cv::rectangle(output, target, color, thickness);
		cv::imshow("Detections", output);
		milliseconds iterationTime = iterationTimer.stop();
		char c = (char)cv::waitKey(pause ? 0 : 5);
		if (c == 'q')
			run = false;
		else if (c == 'p')
			pause = !pause;
		else if (c == 'r')
			tracker.reset();
		else if (c == 's')
			showParticles = !showParticles;
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
