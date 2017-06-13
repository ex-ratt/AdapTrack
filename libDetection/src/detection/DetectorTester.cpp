/*
 * DetectorTester.cpp
 *
 *  Created on: 23.10.2015
 *      Author: poschmann
 */

#include "detection/DetectorTester.hpp"
#include <fstream>
#include <string>

using cv::Mat;
using cv::Rect;
using detection::Detector;
using imageio::AnnotatedImage;
using imageio::Annotation;
using imageio::Annotations;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;
using std::function;
using std::ifstream;
using std::ofstream;
using std::vector;

using namespace std;

namespace detection {

DetectorTester::DetectorTester(cv::Size minWindowSize, double overlapThreshold) :
		minWindowSize(minWindowSize), overlapThreshold(overlapThreshold) {
	if (overlapThreshold <= 0 || overlapThreshold > 1)
		throw invalid_argument("the overlap threshold must be between zero (exclusive) and one (inclusive)");
}

void DetectorTester::evaluate(Detector& detector, const vector<AnnotatedImage>& images) {
	for (const AnnotatedImage& image : images)
		evaluate(detector, image.image, image.annotations);
}

void DetectorTester::evaluate(Detector& detector, const Mat& image, Annotations annotations) {
	ignoreSmallAnnotations(annotations);
	steady_clock::time_point start = steady_clock::now();
	vector<pair<Rect, float>> detections = detector.detectWithScores(image);
	steady_clock::time_point end = steady_clock::now();
	detectionTimeSum += duration_cast<milliseconds>(end - start);
	imageCount += 1;
	positiveCount += annotations.positiveCount();
	mergeInto(classifiedScores, classifyScores(detections, annotations));
}

vector<pair<float, bool>> DetectorTester::classifyScores(const vector<pair<Rect, float>>& detections, Annotations annotations) const {
	vector<pair<float, bool>> classifiedScores;
	classifiedScores.reserve(detections.size());
	for (pair<Rect, float> detection : detections) {
		pair<double, vector<Annotation>::const_iterator> bestMatch = getBestMatch(detection.first, annotations.annotations);
		if (bestMatch.first < overlapThreshold) { // detection is false positive
			classifiedScores.emplace_back(detection.second, false);
		} else {
			if (!bestMatch.second->fuzzy) // detection is true positive
				classifiedScores.emplace_back(detection.second, true);
			annotations.annotations.erase(bestMatch.second);
		}
	}
	return classifiedScores;
}

pair<double, vector<Annotation>::const_iterator> DetectorTester::getBestMatch(Rect detection, const vector<Annotation>& annotations) const {
	if (annotations.empty())
		return make_pair(0.0, annotations.end());
	auto bestAnnotation = annotations.begin();
	double bestOverlap = computeOverlap(detection, bestAnnotation->bounds);
	for (auto it = bestAnnotation + 1; it != annotations.end(); ++it) {
		double overlap = computeOverlap(detection, it->bounds);
		if (overlap > bestOverlap) {
			bestAnnotation = it;
			bestOverlap = overlap;
		}
	}
	return make_pair(bestOverlap, bestAnnotation);
}

void DetectorTester::mergeInto(vector<pair<float, bool>>& scores, const vector<pair<float, bool>>& additionalScores) const {
	vector<pair<float, bool>> mergedScores;
	mergedScores.reserve(scores.size() + additionalScores.size());
	std::merge(scores.begin(), scores.end(),
			additionalScores.begin(), additionalScores.end(),
			std::back_inserter(mergedScores), std::greater<pair<float, bool>>());
	scores.swap(mergedScores);
}

DetectionResult DetectorTester::detect(Detector& detector, const Mat& image, Annotations annotations) const {
	vector<Rect> detections = detector.detect(image);
	ignoreSmallAnnotations(annotations);
	Status status = compareWithGroundTruth(detections, annotations);
	DetectionResult result;
	for (int i = 0; i < detections.size(); ++i) {
		DetectionStatus s = status.detectionStatus[i];
		if (s == DetectionStatus::TRUE_POSITIVE)
			result.correctDetections.push_back(detections[i]);
		else if (s == DetectionStatus::FALSE_POSITIVE)
			result.wrongDetections.push_back(detections[i]);
		else if (s == DetectionStatus::IGNORED)
			result.ignoredDetections.push_back(detections[i]);
	}
	std::vector<cv::Rect> positives = annotations.positiveAnnotations();
	for (int i = 0; i < positives.size(); ++i) {
		PositiveStatus s = status.positiveStatus[i];
		if (s == PositiveStatus::FALSE_NEGATIVE)
			result.missedDetections.push_back(positives[i]);
	}
	return result;
}

void DetectorTester::ignoreSmallAnnotations(Annotations& annotations) const {
	for (Annotation& annotation : annotations.annotations) {
		Rect bounds = annotation.bounds;
		if (bounds.width < minWindowSize.width && bounds.height < minWindowSize.height)
			annotation.fuzzy = true;
	}
}

DetectorTester::Status DetectorTester::compareWithGroundTruth(const vector<Rect>& detections, const Annotations& annotations) const {
	DetectorTester::Status status;
	status.detectionStatus.resize(detections.size());
	status.positiveStatus.resize(annotations.positiveCount());
	Mat positiveOverlaps = createOverlapMatrix(detections, annotations.positiveAnnotations());
	Mat ignoreOverlaps = createOverlapMatrix(detections, annotations.fuzzyAnnotations());
	computeStatus(positiveOverlaps, ignoreOverlaps, status);
	return status;
}

Mat DetectorTester::createOverlapMatrix(const vector<Rect>& detections, const vector<Rect>& annotations) const {
	Mat overlaps(detections.size(), annotations.size(), CV_64FC1);
	for (int row = 0; row < detections.size(); ++row) {
		for (int col = 0; col < annotations.size(); ++col)
			overlaps.at<double>(row, col) = computeOverlap(detections[row], annotations[col]);
	}
	return overlaps;
}

double DetectorTester::computeOverlap(Rect a, Rect b) const {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}

void DetectorTester::computeStatus(Mat& positiveOverlaps, Mat& ignoreOverlaps, DetectorTester::Status& status) const {
	double maxPositiveOverlap, maxIgnoreOverlap;
	cv::Point positiveIndices, ignoreIndices;
	cv::minMaxLoc(positiveOverlaps, nullptr, &maxPositiveOverlap, nullptr, &positiveIndices);
	cv::minMaxLoc(ignoreOverlaps, nullptr, &maxIgnoreOverlap, nullptr, &ignoreIndices);
	while (std::max(maxPositiveOverlap, maxIgnoreOverlap) >= overlapThreshold) {
		if (maxPositiveOverlap >= maxIgnoreOverlap) {
			status.detectionStatus[positiveIndices.y] = DetectionStatus::TRUE_POSITIVE;
			status.positiveStatus[positiveIndices.x] = PositiveStatus::TRUE_POSITIVE;
			positiveOverlaps(cv::Range::all(), cv::Range(positiveIndices.x, positiveIndices.x + 1)) = 0.0;
			positiveOverlaps(cv::Range(positiveIndices.y, positiveIndices.y + 1), cv::Range::all()) = 0.0;
			ignoreOverlaps(cv::Range(positiveIndices.y, positiveIndices.y + 1), cv::Range::all()) = 0.0;
		} else {
			status.detectionStatus[ignoreIndices.y] = DetectionStatus::IGNORED;
			ignoreOverlaps(cv::Range::all(), cv::Range(ignoreIndices.x, ignoreIndices.x + 1)) = 0.0;
			ignoreOverlaps(cv::Range(ignoreIndices.y, ignoreIndices.y + 1), cv::Range::all()) = 0.0;
			positiveOverlaps(cv::Range(ignoreIndices.y, ignoreIndices.y + 1), cv::Range::all()) = 0.0;
		}
		cv::minMaxLoc(positiveOverlaps, nullptr, &maxPositiveOverlap, nullptr, &positiveIndices);
		cv::minMaxLoc(ignoreOverlaps, nullptr, &maxIgnoreOverlap, nullptr, &ignoreIndices);
	}
}

DetectorEvaluationSummary DetectorTester::getSummary() const {
	DetectorEvaluationSummary summary;
	if (imageCount > 0) {
		summary.avgTime = detectionTimeSum / imageCount;
		summary.fps = 1000.0 * imageCount / detectionTimeSum.count();
	}
	bool defaultThresholdFound = false;
	array<double, 9> fppiRates = {
			std::pow(10, -2.00),
			std::pow(10, -1.75),
			std::pow(10, -1.50),
			std::pow(10, -1.25),
			std::pow(10, -1.00),
			std::pow(10, -0.75),
			std::pow(10, -0.50),
			std::pow(10, -0.25),
			std::pow(10,  0.00)
	};
	array<double, 9> missRates;
	array<double, 9> scoreThresholds;
	int nextFppiIndex = 0;
	int truePositives = 0;
	int falsePositives = 0;
	double fppiRate = 0.0;
	double missRate = 1.0;
	float scoreThreshold = std::numeric_limits<double>::infinity();
	for (pair<float, bool> classifiedScore : classifiedScores) {
		float score = classifiedScore.first;
		if (!defaultThresholdFound && score < 0) {
			defaultThresholdFound = true;
			summary.defaultFppiRate = fppiRate;
			summary.defaultMissRate = missRate;
		}
		if (classifiedScore.second)
			++truePositives;
		else
			++falsePositives;
		int falseNegatives = positiveCount - truePositives;
		fppiRate = static_cast<double>(falsePositives) / static_cast<double>(imageCount);
		if (nextFppiIndex < fppiRates.size() && fppiRate > fppiRates[nextFppiIndex]) {
			missRates[nextFppiIndex] = missRate;
			scoreThresholds[nextFppiIndex] = scoreThreshold;
			++nextFppiIndex;
		}
		missRate = static_cast<double>(falseNegatives) / static_cast<double>(positiveCount);
		scoreThreshold = score;
	}
	if (nextFppiIndex > 0) {
		summary.missRateAtFppi2 = missRates[0];
		summary.thresholdAtFppi2 = scoreThresholds[0];
	}
	if (nextFppiIndex > 4) {
		summary.missRateAtFppi1 = missRates[4];
		summary.thresholdAtFppi1 = scoreThresholds[4];
	}
	if (nextFppiIndex > 8) {
		summary.missRateAtFppi0 = missRates[8];
		summary.thresholdAtFppi0 = scoreThresholds[8];
	}
	for (int i = nextFppiIndex; i < missRates.size(); ++i)
		missRates[i] = missRate;
	double missRateSum = 0;
	for (int i = 0; i < missRates.size(); ++i)
		missRateSum += missRates[i];
	summary.avgMissRate = missRateSum / missRates.size();
	return summary;
}

void DetectorTester::storeData(const string& filename) const {
	ofstream file(filename);
	file << "Threshold " << overlapThreshold << '\n';
	file << "Images " << imageCount << '\n';
	file << "Positives " << positiveCount << '\n';
	file << "Time " << detectionTimeSum.count() << '\n';
	file << "Scores\n";
	for (const pair<float, bool>& classifiedScore : classifiedScores)
		file << classifiedScore.first << " " << classifiedScore.second << '\n';
	file.close();
}

void DetectorTester::loadData(const string& filename) {
	ifstream file(filename);
	string tmp;
	file >> tmp >> overlapThreshold; // "Threshold"
	file >> tmp >> imageCount; // "Images"
	file >> tmp >> positiveCount; // "Positives"
	milliseconds::rep timeInMilliseconds;
	file >> tmp >> timeInMilliseconds; // "Time"
	detectionTimeSum = milliseconds(timeInMilliseconds);
	file >> tmp; // "Scores"
	classifiedScores.clear();
	while (!file.eof()) {
		float score;
		bool truePositive;
		file >> score >> truePositive;
		classifiedScores.emplace_back(score, truePositive);
	}
	file.close();
}

void DetectorTester::writePrecisionRecallCurve(const string& filename) const {
	auto recall = [this](int truePositives, int falsePositives) {
		int relevant = positiveCount;
		return static_cast<double>(truePositives) / static_cast<double>(relevant);
	};
	auto precision = [](int truePositives, int falsePositives) {
		int selected = truePositives + falsePositives;
		return static_cast<double>(truePositives) / static_cast<double>(selected);
	};
	writeCurve(filename, recall, precision);
}

void DetectorTester::writeRocCurve(const string& filename, bool writeFalsePositivesPerImage) const {
	auto falsePositives = [](int truePositives, int falsePositives) {
		return falsePositives;
	};
	auto falsePositivesPerImage = [this](int truePositives, int falsePositives) {
		return static_cast<double>(falsePositives) / static_cast<double>(imageCount);
	};
	auto truePositiveRate = [this](int truePositives, int falsePositives) {
		return static_cast<double>(truePositives) / static_cast<double>(positiveCount);
	};
	if (writeFalsePositivesPerImage)
		writeCurve(filename, falsePositivesPerImage, truePositiveRate);
	else
		writeCurve(filename, falsePositives, truePositiveRate);
}

void DetectorTester::writeDetCurve(const string& filename, bool writeFalsePositivesPerImage) const {
	auto falsePositives = [](int truePositives, int falsePositives) {
		return falsePositives;
	};
	auto falsePositivesPerImage = [this](int truePositives, int falsePositives) {
		return static_cast<double>(falsePositives) / static_cast<double>(imageCount);
	};
	auto missedDetectionRate = [this](int truePositives, int falsePositives) {
		int falseNegatives = positiveCount - truePositives;
		return static_cast<double>(falseNegatives) / static_cast<double>(positiveCount);
	};
	if (writeFalsePositivesPerImage)
		writeCurve(filename, falsePositivesPerImage, missedDetectionRate);
	else
		writeCurve(filename, falsePositives, missedDetectionRate);
}

void DetectorTester::writeCurve(string filename, function<double(int, int)> x, function<double(int, int)> y) const {
	ofstream curveFile(filename);
	int truePositives = 0;
	int falsePositives = 0;
	for (pair<float, bool> classifiedScore : classifiedScores) {
		if (classifiedScore.second)
			++truePositives;
		else
			++falsePositives;
		curveFile
				<< std::to_string(x(truePositives, falsePositives)) << ' '
				<< std::to_string(y(truePositives, falsePositives)) << ' '
				<< std::to_string(classifiedScore.first) << '\n';
	}
	curveFile.close();
}

} /* namespace detection */
