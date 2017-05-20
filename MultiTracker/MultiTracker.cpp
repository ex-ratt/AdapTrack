/*
 * MultiTracker.cpp
 *
 *  Created on: 26.10.2016
 *      Author: poschmann
 */

#include "stacktrace.hpp"
#include "StopWatch.hpp"
#include "boost/filesystem.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "detection/AggregatedFeaturesDetector.hpp"
#include "detection/NonMaximumSuppression.hpp"
#include "imageio/CameraImageSource.hpp"
#include "imageio/VideoImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
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

using namespace boost::filesystem;
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

shared_ptr<ImageSource> loadImages(const string& video);
shared_ptr<ProbabilisticSvmClassifier> loadSvm(const string& filename, float threshold = 0);
shared_ptr<FhogFilter> createFhogFilter(int binCount, int cellSize);
shared_ptr<AggregatedFeaturesDetector> createDetector(
		shared_ptr<FhogFilter> fhogFilter, shared_ptr<SvmClassifier> svm, int cellSize, int minWidth, int maxWidth);
void run(MultiTracker& tracker, ImageSource& images);
void drawParticles(Mat& output, vector<tracking::filtering::Particle> particles);

int main(int argc, char **argv) {
	if (argc != 6) {
		cout << "usage: " << argv[0] << " video svm cellsize detectionThreshold visibilityThreshold" << endl;
		cout << "  video: camera device ID, video file, dlib annotation XML-file, or image directory" << endl;
		cout << "  svm: text file that contains SVM data (e.g. created by DetectorTrainer)" << endl;
		cout << "  cellsize: size of the square FHOG cells in pixels" << endl;
		cout << "  detectionThreshold: SVM score threshold for detections to be reported" << endl;
		cout << "  visibilityThreshold: SVM score threshold for tracks to be regarded visible" << endl;
		return EXIT_FAILURE;
	}
	string video = argv[1];
	string svmFile = argv[2];
	int cellSize = std::stoi(argv[3]);
	float detectionThreshold = std::stof(argv[4]);
	float visibilityThreshold = std::stof(argv[5]);
	int minWidth = 0;
	int maxWidth = 0;

	shared_ptr<ImageSource> images = loadImages(video);
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
	run(*tracker, *images);

	return EXIT_SUCCESS;
}

shared_ptr<ImageSource> loadImages(const string& s) {
	if (s.length() == 1 && isdigit(s[0]))
		return make_shared<CameraImageSource>(stoi(s));
	path p(s);
	if (!exists(p))
		throw invalid_argument(s + " is not a valid file or directory");
	if (is_directory(p))
		return make_shared<DirectoryImageSource>(p.string());
	if (".xml" == p.extension().string())
		return make_shared<DlibImageSource>(p.string());
	return make_shared<VideoImageSource>(p.string());
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

void run(MultiTracker& tracker, ImageSource& images) {
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
		Mat frame = images.getImage();
		StopWatch iterationTimer = StopWatch::start();
		vector<pair<int, Rect>> targets = tracker.update(grayscaleFilter.applyTo(frame));
		milliseconds iterationTime = iterationTimer.stop();
		frame.copyTo(output);
		if (debug) {
			for (const Track& track : tracker.getTracks()) {
				Mat intermediate = output.clone();
				if (!track.confirmed)
					rectangle(intermediate, track.state.bounds(), colorUnconfirmed, thickness);
				output = 0.75 * output + 0.25 * intermediate;
			}
			for (const Track& track : tracker.getTracks())
				drawParticles(output, track.filter->getParticles());
		}
		for (const pair<int, Rect>& target : targets)
			rectangle(output, target.second, colors[target.first % colors.size()], thickness);
		imshow("Detections", output);

		iterationTimeSum += iterationTime;
		double iterationFps = static_cast<double>(frameCount) / iterationTimeSum.count();
		cout << fixed << setprecision(1);
		cout << frameCount << ": ";
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

void drawParticles(Mat& output, vector<tracking::filtering::Particle> particles) {
	if (particles.empty())
		return;
	Mat intermediate = output.clone();
	sort(particles.begin(), particles.end(), [](const auto& a, const auto& b) {
		return a.weight < b.weight;
	});
	double maxWeight = particles.back().weight;
	double scale = 1.0 / maxWeight;
	for (const auto& particle : particles) {
		double weight = scale * particle.weight;
		circle(intermediate, Point(particle.state.x, particle.state.y), 3, Scalar(weight * 255, weight * 255, weight * 255), 1);
	}
	output = 0.5 * output + 0.5 * intermediate;
}
