/*
 * DetectorTester.hpp
 *
 *  Created on: 23.10.2015
 *      Author: poschmann
 */

#ifndef DETECTION_DETECTORTESTER_HPP_
#define DETECTION_DETECTORTESTER_HPP_

#include "imageio/AnnotatedImage.hpp"
#include "detection/Detector.hpp"
#include "opencv2/core/core.hpp"
#include <chrono>
#include <vector>

namespace detection {

/**
 * Summary of a detector evaluation.
 */
struct DetectorEvaluationSummary {
	double defaultMissRate = std::numeric_limits<double>::quiet_NaN(); ///< Miss rate using the default threshold of zero.
	double defaultFppiRate = std::numeric_limits<double>::quiet_NaN(); ///< False positive per image rate using the default threshold of zero.
	double missRateAtFppi0 = std::numeric_limits<double>::quiet_NaN(); ///< Miss rate at a false positive per image rate of 1.
	double missRateAtFppi1 = std::numeric_limits<double>::quiet_NaN(); ///< Miss rate at a false positive per image rate of 0.1.
	double missRateAtFppi2 = std::numeric_limits<double>::quiet_NaN(); ///< Miss rate at a false positive per image rate of 0.01.
	double thresholdAtFppi0 = std::numeric_limits<double>::quiet_NaN(); ///< SVM threshold at a false positive per image rate of 1.
	double thresholdAtFppi1 = std::numeric_limits<double>::quiet_NaN(); ///< SVM threshold at a false positive per image rate of 0.1.
	double thresholdAtFppi2 = std::numeric_limits<double>::quiet_NaN(); ///< SVM threshold at a false positive per image rate of 0.01.
	double avgMissRate = std::numeric_limits<double>::quiet_NaN(); ///< Log-average miss rate.
	std::chrono::milliseconds avgTime; ///< Average detection time per image.
	double fps = std::numeric_limits<double>::quiet_NaN(); ///< Detection speed in frames per second.

	/**
	 * Writes the summary data into a stream.
	 *
	 * @param[in] out Stream to write the data into.
	 */
	void writeTo(std::ostream& out) {
		out << "Speed: " << fps << " frames / second" << std::endl;
		out << "Average time: " << avgTime.count() << " ms" << std::endl;
		out << "Default FPPI rate: " << defaultFppiRate << std::endl;
		out << "Default miss rate: " << defaultMissRate << std::endl;
		out << "Miss rate at 1 FPPI: " << missRateAtFppi0 << " (threshold " << thresholdAtFppi0 << ")" << std::endl;
		out << "Miss rate at 0.1 FPPI: " << missRateAtFppi1 << " (threshold " << thresholdAtFppi1 << ")" << std::endl;
		out << "Miss rate at 0.01 FPPI: " << missRateAtFppi2 << " (threshold " << thresholdAtFppi2 << ")" << std::endl;
		out << "Log-average miss rate: " << avgMissRate << std::endl;
	}
};

/**
 * Result of a detection in an image after comparison with the ground truth.
 */
struct DetectionResult {
	std::vector<cv::Rect> correctDetections; ///< True positive detections.
	std::vector<cv::Rect> wrongDetections; ///< False positive detections.
	std::vector<cv::Rect> missedDetections; ///< False negative detections.
	std::vector<cv::Rect> ignoredDetections; ///< Ignored detections.
};

/**
 * Tester for detectors.
 */
class DetectorTester {
public:

	/**
	 * Constructs a new detector tester.
	 *
	 * @param[in] minWindowSize Smallest window size that is tested in pixels.
	 * @param[in] overlapThreshold Minimum overlap necessary to assign a detection to a ground truth bounding box.
	 */
	explicit DetectorTester(cv::Size minWindowSize = cv::Size(), double overlapThreshold = 0.5);

	/**
	 * Detects targets inside an image using a detector.
	 *
	 * @param[in] detector Detector.
	 * @param[in] image Image.
	 * @param[in] annotations Annotated bounding boxes.
	 * @return Result containing correct, wrong, ignored and missed detections.
	 */
	DetectionResult detect(detection::Detector& detector, const cv::Mat& image, imageio::Annotations annotations) const;

	/**
	 * Evaluates a detector on several images.
	 *
	 * @param[in] detector Detector that should be evaluated.
	 * @param[in] images Images with annotated bounding boxes.
	 */
	void evaluate(detection::Detector& detector, const std::vector<imageio::AnnotatedImage>& images);

	/**
	 * Evaluates a detector on a single image.
	 *
	 * @param[in] detector Detector that should be evaluated.
	 * @param[in] image Image to detect targets in.
	 * @param[in] annotations Annotated bounding boxes.
	 */
	void evaluate(detection::Detector& detector, const cv::Mat& image, imageio::Annotations annotations);

	/**
	 * @return Summary of the evaluation.
	 */
	DetectorEvaluationSummary getSummary() const;

	/**
	 * Stores the evaluation data into a file.
	 *
	 * @param[in] filename Name of the file to store the data into.
	 */
	void storeData(const std::string& filename) const;

	/**
	 * Loads evaluation data from a file, replacing the current data of this tester.
	 *
	 * @param[in] filename Name of the file to load the data from.
	 */
	void loadData(const std::string& filename);

	/**
	 * Writes the points of the precision recall curve into the given file.
	 *
	 * @param[in] filename Name of the file to write the points into.
	 */
	void writePrecisionRecallCurve(const std::string& filename) const;

	/**
	 * Writes the points of the ROC curve into the given file.
	 *
	 * @param[in] filename Name of the file to write the points into.
	 * @param[in] falsePositivesPerImage Flag that indicates whether to write false positives per image instead of false positives.
	 */
	void writeRocCurve(const std::string& filename, bool falsePositivesPerImage = true) const;

	/**
	 * Writes the points of the DET curve into the given file.
	 *
	 * @param[in] filename Name of the file to write the points into.
	 * @param[in] falsePositivesPerImage Flag that indicates whether to write false positives per image instead of false positives.
	 */
	void writeDetCurve(const std::string& filename, bool falsePositivesPerImage = true) const;

private:

	enum class DetectionStatus : uint8_t {
		FALSE_POSITIVE, TRUE_POSITIVE, IGNORED
	};

	enum class PositiveStatus : uint8_t {
		FALSE_NEGATIVE, TRUE_POSITIVE
	};

	struct Status {
		std::vector<DetectionStatus> detectionStatus;
		std::vector<PositiveStatus> positiveStatus;
	};

	/**
	 * Classifies the detection scores as either true positive or false positive.
	 *
	 * @param[in] detection Detected objects ordered by their score in descending order.
	 * @param[in] annotations Annotated objects.
	 * @return Detections scores in descending order with their binary classification label.
	 */
	std::vector<std::pair<float, bool>> classifyScores(
			const std::vector<std::pair<cv::Rect, float>>& detections, imageio::Annotations annotations) const;

	/**
	 * Determines the annotation that overlaps the most with a detection.
	 *
	 * @param[in] detection Detected object.
	 * @param[in] annotations Annotated objects.
	 * @return Overlap ratio and iterator to the best matching annotation.
	 */
	std::pair<double, std::vector<imageio::Annotation>::const_iterator> getBestMatch(
			cv::Rect detection, const std::vector<imageio::Annotation>& annotations) const;

	/**
	 * Merges additional classified scores into an existing vector of classified scores.
	 *
	 * @param[in] scores Existing classified scores.
	 * @param[in] additionalScores Additional classified scores that should be merged into the existing vector.
	 */
	void mergeInto(std::vector<std::pair<float, bool>>& scores, const std::vector<std::pair<float, bool>>& additionalScores) const;

	void ignoreSmallAnnotations(imageio::Annotations& annotations) const;

	Status compareWithGroundTruth(const std::vector<cv::Rect>& detections, const imageio::Annotations& annotations) const;

	cv::Mat createOverlapMatrix(const std::vector<cv::Rect>& detections, const std::vector<cv::Rect>& annotations) const;

	double computeOverlap(cv::Rect a, cv::Rect b) const;

	void computeStatus(cv::Mat& positiveOverlaps, cv::Mat& ignoreOverlaps, DetectorTester::Status& status) const;

	void writeCurve(std::string filename, std::function<double(int, int)> x, std::function<double(int, int)> y) const;

	cv::Size minWindowSize; ///< Smallest window size that is tested in pixels.
	double overlapThreshold; ///< Minimum overlap necessary to assign a detection to a ground truth bounding box.
	int imageCount = 0; ///< Number of evaluated images.
	int positiveCount = 0; ///< Number of positive annotations.
	std::vector<std::pair<float, bool>> classifiedScores; ///< Detection scores with flag that indicates whether the detection was a true positive.
	std::chrono::milliseconds detectionTimeSum = std::chrono::milliseconds::zero(); ///< Sum of detection times.
};

} /* namespace detection */

#endif /* DETECTION_DETECTORTESTER_HPP_ */
