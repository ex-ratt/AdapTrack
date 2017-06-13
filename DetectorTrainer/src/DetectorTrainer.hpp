/*
 * DetectorTrainer.hpp
 *
 *  Created on: 21.10.2015
 *      Author: poschmann
 */

#ifndef DETECTORTRAINER_HPP_
#define DETECTORTRAINER_HPP_

#include "classification/ClassifierTrainer.hpp"
#include "classification/ConfidenceBasedExampleManagement.hpp"
#include "classification/ExampleManagement.hpp"
#include "classification/ProbabilisticSupportVectorMachine.hpp"
#include "classification/SupportVectorMachine.hpp"
#include "detection/AggregatedFeaturesDetector.hpp"
#include "detection/NonMaximumSuppression.hpp"
#include "imageio/AnnotatedImage.hpp"
#include "imageprocessing/extraction/AggregatedFeaturesExtractor.hpp"
#include "opencv2/core/core.hpp"
#include <memory>
#include <random>
#include <string>
#include <vector>

/**
 * Management of negative training examples that only keeps the hardest examples.
 */
class HardNegativeExampleManagement : public classification::ConfidenceBasedExampleManagement {
public:

	HardNegativeExampleManagement(const std::shared_ptr<classification::BinaryClassifier>& classifier, size_t capacity) :
			classification::ConfidenceBasedExampleManagement(classifier, false, capacity) {
		setFirstExamplesToKeep(0);
	}
};

/**
 * Trainer for detectors based on aggregated features and a linear support vector machine.
 */
class DetectorTrainer {
public:

	/**
	 * @param[in] featureExtractor Extractor of the aggregated features.
	 */
	void setFeatureExtractor(std::shared_ptr<imageprocessing::extraction::AggregatedFeaturesExtractor> featureExtractor);

	/**
	 * Sets the trainer of the support vector machine.
	 *
	 * @param[in] trainer Trainer of the detector's support vector machine.
	 */
	void setSvmTrainer(std::shared_ptr<classification::ClassifierTrainer<classification::SupportVectorMachine>> trainer);

	/**
	 * Sets trainer of the probabilistic support vector machine. The detector ignores the probabilistic properties,
	 * but they are stored with the classifier when calling storeClassifier(std::string).
	 *
	 * @param[in] trainer Trainer of the detector's support vector machine.
	 */
	void setProbabilisticSvmTrainer(
			std::shared_ptr<classification::ClassifierTrainer<classification::ProbabilisticSupportVectorMachine>> trainer);

	/**
	 * Trains the classifier that is used by the detector.
	 *
	 * The labeled images contain positive examples and fuzzy ones that will be ignored for training. Fuzzy training
	 * examples must have a name that starts with "ignore". Bounding boxes with other names are considered positive.
	 *
	 * @param[in] images Images labeled with bounding boxes around positive and fuzzy examples (anything else is considered negative).
	 */
	void train(std::vector<imageio::AnnotatedImage> images);

	/**
	 * Stores the SVM data into a file.
	 *
	 * @param[in] filename Name of the file.
	 */
	void storeClassifier(const std::string& filename) const;

	/**
	 * @return Weight vector of the SVM.
	 */
	cv::Mat getWeightVector() const;

	/**
	 * Creates a new detector that uses the trained classifier.
	 *
	 * @param[in] nms Non-maximum suppression algorithm.
	 */
	std::shared_ptr<detection::AggregatedFeaturesDetector> getDetector(
			std::shared_ptr<detection::NonMaximumSuppression> nms) const;

	/**
	 * Creates a new detector that uses the trained classifier, but uses a different feature extractor than was used
	 * for training.
	 *
	 * @param[in] nms Non-maximum suppression algorithm.
	 * @param[in] featureExtractor Feature extractor.
	 * @param[in] threshold SVM score threshold, defaults to zero.
	 */
	std::shared_ptr<detection::AggregatedFeaturesDetector> getDetector(std::shared_ptr<detection::NonMaximumSuppression> nms,
			std::shared_ptr<imageprocessing::extraction::AggregatedFeaturesExtractor> featureExtractor, float threshold = 0) const;

private:

	void createEmptyClassifier();

	void collectInitialTrainingExamples(std::vector<imageio::AnnotatedImage> images);

	void collectHardTrainingExamples(std::vector<imageio::AnnotatedImage> images);

	void createHardNegativesDetector();

	void collectTrainingExamples(std::vector<imageio::AnnotatedImage> images, bool initial);

	/**
	 * Adjusts the size and aspect ratio of the annotations to fit the feature window size.
	 *
	 * @param[in] annotations Annotated bounding boxes.
	 * @return Annotated bounding boxes with the correct aspect ratio.
	 */
	imageio::Annotations adjustSizes(const imageio::Annotations& annotations) const;

	/**
	 * Adjusts the size and aspect ratio of an annotation to fit the feature window size.
	 *
	 * @param[in] annotation Bounding box with potentially differing aspect ratio.
	 * @return Bounding box with the correct aspect ratio.
	 */
	imageio::Annotation adjustSize(imageio::Annotation annotation) const;

	void addMirroredTrainingExamples(const cv::Mat& image, const imageio::Annotations& annotations, bool initial);

	cv::Mat flipHorizontally(const cv::Mat& image);

	imageio::Annotations flipHorizontally(const imageio::Annotations& annotations, int imageWidth);

	imageio::Annotation flipHorizontally(imageio::Annotation annotation, int imageWidth);

	void addTrainingExamples(const cv::Mat& image, const imageio::Annotations& annotations, bool initial);

	void setImage(const cv::Mat& image);

	void addPositiveExamples(const std::vector<cv::Rect>& positiveBoxes);

	void addRandomNegativeExamples(const std::vector<cv::Rect>& nonNegativeBoxes);

	cv::Rect createRandomBounds() const;

	void addHardNegativeExamples(const std::vector<cv::Rect>& nonNegativeBoxes);

	bool addNegativeIfNotOverlapping(cv::Rect candidate, const std::vector<cv::Rect>& nonNegativeBoxes);

	bool isOverlapping(cv::Rect boxToTest, const std::vector<cv::Rect>& otherBoxes) const;

	double computeOverlap(cv::Rect a, cv::Rect b) const;

	void trainClassifier();

	void retrainClassifier();

	void trainSvm();

public:

	bool printProgressInformation = false; ///< Flag that indicates whether to print progress information to cout.
	std::string printPrefix = ""; ///< Prefix that is printed before each line of progress information.
	bool mirrorTrainingData = true; ///< Flag that indicates whether to horizontally mirror the training data.
	int maxNegatives = 0; ///< Maximum number of negative training examples (0 if not constrained).
	int randomNegativesPerImage = 20; ///< Number of initial random negatives per image.
	int maxHardNegativesPerImage = 100; ///< Maximum number of hard negatives per image and bootstrapping round.
	int bootstrappingRounds = 3; ///< Number of bootstrapping rounds.
	float negativeScoreThreshold = -1.0f; ///< SVM score threshold for retrieving strong negative examples.
	double overlapThreshold = 0.3; ///< Maximum allowed overlap between negative examples and non-negative annotations.

private:

	mutable std::mt19937 generator = std::mt19937(std::random_device()());
	double aspectRatio = 1;
	double aspectRatioInv = 1;
	std::shared_ptr<detection::NonMaximumSuppression> noSuppression = std::make_shared<detection::NonMaximumSuppression>(1.0);
	std::shared_ptr<imageprocessing::extraction::AggregatedFeaturesExtractor> featureExtractor;
	std::shared_ptr<classification::SupportVectorMachine> svm;
	std::shared_ptr<classification::ProbabilisticSupportVectorMachine> probabilisticSvm;
	std::shared_ptr<classification::ClassifierTrainer<classification::SupportVectorMachine>> svmTrainer;
	std::shared_ptr<classification::ClassifierTrainer<classification::ProbabilisticSupportVectorMachine>> probabilisticSvmTrainer;
	std::shared_ptr<detection::AggregatedFeaturesDetector> hardNegativesDetector;
	std::unique_ptr<classification::ExampleManagement> positives;
	std::unique_ptr<classification::ExampleManagement> negatives;
	std::vector<cv::Mat> newPositives;
	std::vector<cv::Mat> newNegatives;
	cv::Mat image;
	cv::Size imageSize;
};

#endif /* DETECTORTRAINER_HPP_ */
