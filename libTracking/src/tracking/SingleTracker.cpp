/*
 * SingleTracker.cpp
 *
 *  Created on: May 11, 2017
 *      Author: ex-ratt
 */

#include "tracking/SingleTracker.hpp"
#include "classification/IncrementalLinearSvmTrainer.hpp"
#include "classification/LinearKernel.hpp"
#include "libsvm/LibSvmTrainer.hpp"
#include <stdexcept>

using namespace classification;
using namespace cv;
using namespace imageprocessing::filtering;
using namespace std;
using libsvm::LibSvmTrainer;

namespace tracking {

SingleTracker::SingleTracker(int binCount, int cellSize, int targetSize, int padding,
		double scaleFactor, double svmC, double adaptationRate) :
				SingleTracker(make_shared<FhogFilter>(binCount, cellSize, false, true, 0.2f),
						targetSize, padding, scaleFactor, svmC, adaptationRate) {}

SingleTracker::SingleTracker(shared_ptr<FhogFilter> fhogFilter,
		int targetSize, int padding, double scaleFactor, double svmC, double adaptationRate) :
		generator(random_device()()),
		fhogFilter(fhogFilter),
		svm(make_shared<SupportVectorMachine>(make_shared<LinearKernel>())),
		svmTrainer(make_shared<IncrementalLinearSvmTrainer>(make_shared<LibSvmTrainer>(svmC, true), adaptationRate)),
		convolutionFilter(make_shared<ConvolutionFilter>(CV_32F)),
		targetSize(targetSize, targetSize),
		windowSize(targetSize + 2 * padding, targetSize + 2 * padding),
		scaleFactor(scaleFactor),
		negativeExampleCount(10),
		negativeOverlapThreshold(0.5) {
	convolutionFilter->setAnchor(Point(0, 0));
	if (targetSize < 1)
		throw invalid_argument("SingleTracker: target size must be greater than zero");
	if (2 * padding < targetSize)
		throw invalid_argument("SingleTracker: padding must be at least half the target size");
	if (scaleFactor <= 1)
		throw invalid_argument("SingleTracker: scale factor must be greater than one");
}

Rect SingleTracker::init(const Mat& image, Rect bounds, bool force) {
	targetSize.width = max(targetSize.width, targetSize.height);
	targetSize.height = max(targetSize.width, targetSize.height);
	windowSize.width = max(windowSize.width, windowSize.height);
	windowSize.height = max(windowSize.width, windowSize.height);
	double cx = bounds.x + 0.5 * bounds.width;
	double cy = bounds.y + 0.5 * bounds.height;
	targetBounds = bounds;
	if (bounds.height > bounds.width) {
		targetSize.width = static_cast<int>(ceil(targetSize.height * bounds.width / static_cast<double>(bounds.height)));
		double width = bounds.height * targetSize.width / static_cast<double>(targetSize.height);
		targetBounds.x = static_cast<int>(round(cx - 0.5 * width));
		targetBounds.width = static_cast<int>(round(width));
		if (targetBounds.width % 2 != windowSize.width % 2)
			--windowSize.width;
	} else if (bounds.width > bounds.height) {
		targetSize.height = static_cast<int>(ceil(targetSize.width * bounds.height / static_cast<double>(bounds.width)));
		double height = bounds.width * targetSize.height / static_cast<double>(targetSize.width);
		targetBounds.y = static_cast<int>(round(cy - 0.5 * height));
		targetBounds.height = static_cast<int>(round(height));
		if (targetBounds.height % 2 != windowSize.height % 2)
			--windowSize.height;
	}
	if (force ||
			(isTargetWithinImageBounds(image, targetBounds) && !isTargetTooSmall(targetBounds.width, targetBounds.height))) {
		Mat window = getFeatures(image, getWindowBounds(image, targetBounds));
		svmTrainer->train(*svm, getPositiveTrainingExamples(window), getNegativeTrainingExamples(window));
		convolutionFilter->setKernel(svm->getSupportVectors()[0]);
		convolutionFilter->setDelta(-svm->getBias());
		svmTrainer->train(*svm, getPositiveTrainingExamples(window), getNegativeTrainingExamples(window, *svm));
		convolutionFilter->setKernel(svm->getSupportVectors()[0]);
		convolutionFilter->setDelta(-svm->getBias());
	} else { // target is (partially) outside image or too small - and initialization is not forced
		targetBounds = Rect();
	}
	return targetBounds;
}

Rect SingleTracker::update(const Mat& image) {
	if (targetBounds.area() == 0)
		throw runtime_error("SingleTracker: not initialized (target bounds too small or outside the image)");
	Rect windowBounds = getWindowBounds(image, targetBounds);
	Mat window = getFeatures(image, windowBounds);
	pair<Point2d, double> maxScore = getMaxScore(window);

	int cx = targetBounds.x + targetBounds.width / 2;
	int cy = targetBounds.y + targetBounds.height / 2;

	int largerWidth = static_cast<int>(ceil(targetBounds.width * scaleFactor));
	int largerHeight = static_cast<int>(ceil(targetBounds.height * scaleFactor));
	if (!isTargetTooBig(image, largerWidth, largerHeight)) {
		Rect largerTargetBounds(cx - largerWidth / 2, cy - largerHeight/ 2, largerWidth, largerHeight);
		Rect largerWindowBounds = getWindowBounds(image, largerTargetBounds);
		Mat largerWindow = getFeatures(image, largerWindowBounds);
		pair<Point2d, double> largerMaxScore = getMaxScore(largerWindow);
		if (largerMaxScore.second > maxScore.second) {
			window = largerWindow;
			windowBounds = largerWindowBounds;
			maxScore = largerMaxScore;
		}
	}

	int smallerWidth = static_cast<int>(floor(targetBounds.width / scaleFactor));
	int smallerHeight = static_cast<int>(floor(targetBounds.height / scaleFactor));
	if (!isTargetTooSmall(smallerWidth, smallerHeight)) {
		Rect smallerTargetBounds(cx - smallerWidth / 2, cy - smallerHeight / 2, smallerWidth, smallerHeight);
		Rect smallerWindowBounds = getWindowBounds(image, smallerTargetBounds);
		Mat smallerWindow = getFeatures(image, smallerWindowBounds);
		pair<Point2d, double> smallerMaxScore = getMaxScore(smallerWindow);
		if (smallerMaxScore.second > maxScore.second) {
			window = smallerWindow;
			windowBounds = smallerWindowBounds;
			maxScore = smallerMaxScore;
		}
	}

	Point2d point = maxScore.first;
	targetBounds.x = static_cast<int>(round(windowBounds.x + point.x * windowBounds.width / window.cols));
	targetBounds.y = static_cast<int>(round(windowBounds.y + point.y * windowBounds.height / window.rows));
	targetBounds.width = static_cast<int>(round(targetSize.width * windowBounds.width / static_cast<double>(window.cols)));
	targetBounds.height = static_cast<int>(round(targetSize.height * windowBounds.height / static_cast<double>(window.rows)));

	cx = targetBounds.x + targetBounds.width / 2;
	cy = targetBounds.y + targetBounds.height / 2;
	if (cx < 0 || cx >= image.cols || cy < 0 || cy >= image.rows) {
		cx = max(cx, 0);
		cx = min(cx, image.cols - 1);
		cy = max(cy, 0);
		cy = min(cy, image.rows - 1);
		targetBounds.x = cx - targetBounds.width / 2;
		targetBounds.y = cy - targetBounds.height / 2;
	}

	if (isTargetWithinImageBounds(image, targetBounds)) {
		window = getFeatures(image, getWindowBounds(image, targetBounds));
		svmTrainer->retrain(*svm, getPositiveTrainingExamples(window), getNegativeTrainingExamples(window, *svm));
		convolutionFilter->setKernel(svm->getSupportVectors()[0]);
		convolutionFilter->setDelta(-svm->getBias());
	}

  return targetBounds;
}

bool SingleTracker::isTargetTooSmall(int width, int height) const {
	return width < fhogFilter->getCellSize() * targetSize.width / 2
			|| height < fhogFilter->getCellSize() * targetSize.height / 2;
}

bool SingleTracker::isTargetTooBig(const Mat& image, int width, int height) const {
	return width > image.cols || height > image.rows;
}

bool SingleTracker::isTargetWithinImageBounds(const Mat& image, Rect targetBounds) const {
	return targetBounds.x >= 0 && targetBounds.x + targetBounds.width < image.cols
			&& targetBounds.y >= 0 && targetBounds.y + targetBounds.height < image.rows;
}

Rect SingleTracker::getWindowBounds(const Mat& image, Rect targetBounds) const {
	double cx = targetBounds.x + 0.5 * targetBounds.width;
	double cy = targetBounds.y + 0.5 * targetBounds.height;
	double height;
	double width;
	if (targetSize.height > targetSize.width) {
		height = targetBounds.height * windowSize.height / static_cast<double>(targetSize.height);
		width = height * windowSize.width / windowSize.height;
	} else {
		width = targetBounds.width * windowSize.width / static_cast<double>(targetSize.width);
		height = width * windowSize.height / windowSize.width;
	}
	return Rect(
			static_cast<int>(round(cx - 0.5 * width)),
			static_cast<int>(round(cy - 0.5 * height)),
			static_cast<int>(round(width)),
			static_cast<int>(round(height)));
}

Mat SingleTracker::getFeatures(const Mat& image, Rect windowBounds) const {
	Mat window;
	if (windowBounds.x < 0 || windowBounds.y < 0
			|| windowBounds.x + windowBounds.width >= image.cols
			|| windowBounds.y + windowBounds.height >= image.rows) {
		double wl = max(0, windowBounds.x);
		double wt = max(0, windowBounds.y);
		double wr = min(image.cols, windowBounds.x + windowBounds.width);
		double wb = min(image.rows, windowBounds.y + windowBounds.height);
		copyMakeBorder(image(Rect(wl, wt, wr - wl, wb - wt)), window,
				wt - windowBounds.y, windowBounds.y + windowBounds.height - wb,
				wl - windowBounds.x, windowBounds.x + windowBounds.width - wr, BORDER_REPLICATE);
	} else {
		window = image(windowBounds);
	}
	Mat resizedWindow;
	int cellSize = fhogFilter->getCellSize();
	Size newSize(windowSize.width * cellSize, windowSize.height * cellSize);
	int interpolation = newSize.width > window.cols ? INTER_LINEAR : INTER_AREA;
	resize(window, resizedWindow, newSize, 0, 0, interpolation);
	return fhogFilter->applyTo(resizedWindow);
}

pair<Point2d, double> SingleTracker::getMaxScore(const Mat& window) const {
	Mat convolvedWindow = convolutionFilter->applyTo(window);
	convolvedWindow = convolvedWindow(Rect(0, 0, window.cols - targetSize.width + 1, window.rows - targetSize.height + 1));
  Point iPoint;
  double maxScore;
	minMaxLoc(convolvedWindow, nullptr, &maxScore, nullptr, &iPoint);
  Point2d point(iPoint.x, iPoint.y);
	if (iPoint.x > 0 && iPoint.x < convolvedWindow.cols - 1) {
		double prevScore = convolvedWindow.at<float>(iPoint.y, iPoint.x - 1);
		double nextScore = convolvedWindow.at<float>(iPoint.y, iPoint.x + 1);
		point.x += subPixelPeak(prevScore, maxScore, nextScore);
	}
	if (iPoint.y > 0 && iPoint.y < convolvedWindow.rows - 1) {
		double prevScore = convolvedWindow.at<float>(iPoint.y - 1, iPoint.x);
		double nextScore = convolvedWindow.at<float>(iPoint.y + 1, iPoint.x);
		point.y += subPixelPeak(prevScore, maxScore, nextScore);
	}
	return make_pair(point, maxScore);
}

vector<Mat> SingleTracker::getPositiveTrainingExamples(const Mat& window) const {
	Rect target((window.cols - targetSize.width) / 2, (window.rows - targetSize.height) / 2, targetSize.width, targetSize.height);
	return vector<Mat>{ window(target).clone() };
}

vector<Mat> SingleTracker::getNegativeTrainingExamples(const Mat& window) const {
	Rect target((window.cols - targetSize.width) / 2, (window.rows - targetSize.height) / 2, targetSize.width, targetSize.height);
	vector<Mat> trainingExamples;
	trainingExamples.reserve(10);
	while (trainingExamples.size() < trainingExamples.capacity()) {
		int x = uniform_int_distribution<int>{0, window.cols - targetSize.width}(generator);
		int y = uniform_int_distribution<int>{0, window.rows - targetSize.height}(generator);
		Rect candidate(x, y, targetSize.width, targetSize.height);
		if (computeOverlap(target, candidate) < negativeOverlapThreshold) {
			Mat example = window(candidate).clone();
			trainingExamples.push_back(example);
		}
	}
	return trainingExamples;
}

vector<Mat> SingleTracker::getNegativeTrainingExamples(const Mat& window, const SupportVectorMachine& svm) const {
	Rect target((window.cols - targetSize.width) / 2, (window.rows - targetSize.height) / 2, targetSize.width, targetSize.height);
	vector<pair<double, Mat>> trainingCandidates;
	trainingCandidates.reserve(3 * negativeExampleCount);
	while (trainingCandidates.size() < trainingCandidates.capacity()) {
		int x = uniform_int_distribution<int>{0, window.cols - targetSize.width}(generator);
		int y = uniform_int_distribution<int>{0, window.rows - targetSize.height}(generator);
		Rect candidate(x, y, targetSize.width, targetSize.height);
		if (computeOverlap(target, candidate) < negativeOverlapThreshold) {
			Mat example = window(candidate).clone();
			double score = svm.computeHyperplaneDistance(example);
			trainingCandidates.emplace_back(score, example);
		}
	}
	std::partial_sort(trainingCandidates.begin(), trainingCandidates.begin() + negativeExampleCount, trainingCandidates.end(),
			[](const pair<double, Mat>& a, const pair<double, Mat>& b) { return a.first > b.first; });
	vector<Mat> trainingExamples;
	trainingExamples.reserve(negativeExampleCount);
	for (int i = 0; i < trainingExamples.capacity(); ++i)
		trainingExamples.push_back(trainingCandidates[i].second);
	return trainingExamples;
}

double SingleTracker::computeOverlap(Rect a, Rect b) const {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}

double SingleTracker::subPixelPeak(double left, double center, double right) const {
	double divisor = 2 * center - right - left;
	if (divisor == 0)
		return 0;
	return 0.5 * (right - left) / divisor;
}

} // namespace tracking
