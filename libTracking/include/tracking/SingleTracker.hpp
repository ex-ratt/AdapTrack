/*
 * SingleTracker.hpp
 *
 *  Created on: May 11, 2017
 *      Author: ex-ratt
 */

#ifndef SINGLETRACKER_HPP_
#define SINGLETRACKER_HPP_

#include "classification/IncrementalClassifierTrainer.hpp"
#include "classification/SvmClassifier.hpp"
#include "imageprocessing/ConvolutionFilter.hpp"
#include "imageprocessing/filtering/FhogFilter.hpp"
#include "opencv2/core/core.hpp"
#include <memory>
#include <random>
#include <utility>

namespace tracking {

/**
 * Adaptive single target tracker.
 *
 * This tracker needs to be initialized with the bounding box indicating the target position
 * in the first frame. For subsequent frames, the tracker predicts the new target positions
 * itself. It is based on FHOG features and a linear support vector machine.
 */
class SingleTracker {
public:

	/**
	 * Constructs a new single target tracker from parameters.
	 *
	 * @param[in] binCount Number of bins of the FHOG descriptor's unsigned gradient histogram.
	 * @param[in] cellSize Size of the square FHOG cells in pixels.
	 * @param[in] targetSize Size of the target in FHOG cells (larger one of width or height).
	 * @param[in] padding Number of cells around the previous target position that is searched for the new position.
	 * @param[in] scaleFactor Scale factor of neighboring scales that are searched for the target.
	 * @param[in] svmC Soft margin parameter of the SVM.
	 * @param[in] adaptationRate Weight of the new SVM parameters (between zero and one).
	 */
	SingleTracker(int binCount = 9, int cellSize = 4, int targetSize = 10, int padding = 7,
			double scaleFactor = 1.05, double svmC = 10, double adaptationRate = 0.01);

	/**
	 * Constructs a new single target tracker from an existing FHOG filter and parameters.
	 *
	 * @param[in] fhogFilter Filter that computes the FHOG descriptors of the search window.
	 * @param[in] targetSize Size of the target in FHOG cells (larger one of width or height).
	 * @param[in] padding Number of cells around the previous target position that is searched for the new position.
	 * @param[in] scaleFactor Scale factor of neighboring scales that are searched for the target.
	 * @param[in] svmC Soft margin parameter of the SVM.
	 * @param[in] adaptationRate Weight of the new SVM parameters (between zero and one).
	 */
	SingleTracker(std::shared_ptr<imageprocessing::filtering::FhogFilter> fhogFilter,
			int targetSize, int padding, double scaleFactor, double svmC, double adaptationRate);

	/**
	 * Initializes the tracker on the first frame and given bounding box that indicates the initial target position. If
	 * the initialization is not forced, then the bounding box must be completely within the image bounds and it must
	 * not be too small. Otherwise, the tracker will remain uninitialized.
	 *
	 * @param[in] image First frame.
	 * @param[in] bounds Bounding box that indicates the initial target position.
	 * @param[in] force Flag that indicates whether to force the initialization, regardless of bounding box size and position.
	 * @return Bounding box around the target position with adjusted aspect ratio to fit the tracker; empty if not initialized.
	 */
	cv::Rect init(const cv::Mat& image, cv::Rect bounds, bool force = true);

	/**
	 * Updates the tracker with a frame and estimates the new target position.
	 *
	 * @param[in] image New frame.
	 * @return Bounding box around the estimated target position.
	 */
	cv::Rect update(const cv::Mat& image);

private:

	bool isTargetTooSmall(int width, int height) const;

	bool isTargetTooBig(const cv::Mat& image, int width, int height) const;

	bool isTargetWithinImageBounds(const cv::Mat& image, cv::Rect targetBounds) const;

	cv::Rect getWindowBounds(const cv::Mat& image, cv::Rect targetBounds) const;

	cv::Mat getFeatures(const cv::Mat& image, cv::Rect windowBounds) const;

	std::pair<cv::Point2d, double> getMaxScore(const cv::Mat& window) const;

	std::vector<cv::Mat> getPositiveTrainingExamples(const cv::Mat& window) const;

	std::vector<cv::Mat> getNegativeTrainingExamples(const cv::Mat& window) const;

	std::vector<cv::Mat> getNegativeTrainingExamples(const cv::Mat& window, const classification::SvmClassifier& svm) const;

	double computeOverlap(cv::Rect a, cv::Rect b) const;

	double subPixelPeak(double left, double center, double right) const;

	mutable std::default_random_engine generator; ///< Random number generator.
	std::shared_ptr<imageprocessing::filtering::FhogFilter> fhogFilter; ///< Filter that computes the FHOG descriptors of the search window.
	std::shared_ptr<classification::SvmClassifier> svm; ///< SVM that is adapted to the target.
	std::shared_ptr<classification::IncrementalClassifierTrainer<classification::SvmClassifier>> svmTrainer; ///< SVM trainer.
	std::shared_ptr<imageprocessing::ConvolutionFilter> convolutionFilter; ///< Filter that convolves the FHOG window with the SVM weight.
	cv::Size targetSize; ///< Size of the target in FHOG cells.
	cv::Size windowSize; ///< Size of the search window in FHOG cells.
	double scaleFactor; ///< Scale factor of neighboring scales that are searched for the target.
	int negativeExampleCount; ///< Number of negative training examples per classifier update.
	double negativeOverlapThreshold; ///< Bounding box overlap ratio threshold of negative training examples with target position.
	cv::Rect targetBounds; //< Current bounding box of the target in pixels.
};

} // namespace tracking

#endif /* SINGLETRACKER_HPP_ */
