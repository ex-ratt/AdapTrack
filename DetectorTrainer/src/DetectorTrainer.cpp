/*
 * DetectorTrainer.cpp
 *
 *  Created on: 21.10.2015
 *      Author: poschmann
 */

#include "DetectorTrainer.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/SvmClassifier.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/Patch.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

using classification::ExampleManagement;
using classification::LinearKernel;
using cv::Mat;
using cv::Rect;
using detection::AggregatedFeaturesDetector;
using detection::NonMaximumSuppression;
using imageio::AnnotatedImage;
using imageio::Annotation;
using imageio::Annotations;
using imageprocessing::ImageFilter;
using imageprocessing::Patch;
using imageprocessing::extraction::AggregatedFeaturesExtractor;
using libsvm::LibSvmClassifier;
using std::make_shared;
using std::runtime_error;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

DetectorTrainer::DetectorTrainer(bool printProgressInformation, std::string printPrefix) :
		printProgressInformation(printProgressInformation),
		printPrefix(printPrefix),
		aspectRatio(1),
		aspectRatioInv(1),
		noSuppression(make_shared<NonMaximumSuppression>(1.0)),
		generator(std::random_device()()) {}

shared_ptr<AggregatedFeaturesDetector> DetectorTrainer::getDetector(shared_ptr<NonMaximumSuppression> nms) const {
	return getDetector(nms, featureParams.octaveLayerCount);
}

shared_ptr<AggregatedFeaturesDetector> DetectorTrainer::getDetector(
		shared_ptr<NonMaximumSuppression> nms, int octaveLayerCount, float threshold) const {
	if (!classifier->isUsable())
		throw runtime_error("DetectorTrainer: must train a classifier first");
	classifier->getSvm()->setThreshold(threshold);
	shared_ptr<AggregatedFeaturesDetector> detector;
	if (!imageFilter)
		detector = make_shared<AggregatedFeaturesDetector>(filter, featureParams.cellSizeInPixels,
				featureParams.windowSizeInCells, octaveLayerCount, classifier->getSvm(), nms,
				featureParams.widthScaleFactorInv(), featureParams.heightScaleFactorInv());
	else
		detector = make_shared<AggregatedFeaturesDetector>(imageFilter, filter, featureParams.cellSizeInPixels,
				featureParams.windowSizeInCells, octaveLayerCount, classifier->getSvm(), nms,
				featureParams.widthScaleFactorInv(), featureParams.heightScaleFactorInv());
	classifier->getSvm()->setThreshold(0);
	return detector;
}

void DetectorTrainer::storeClassifier(const string& filename) const {
	std::ofstream stream(filename);
	if (trainingParams.probabilistic)
		classifier->getProbabilisticSvm()->store(stream);
	else
		classifier->getSvm()->store(stream);
	stream.close();
}

Mat DetectorTrainer::getWeightVector() const {
	return classifier->getSvm()->getSupportVectors().front();
}

void DetectorTrainer::setTrainingParameters(TrainingParams params) {
	trainingParams = params;
}

void DetectorTrainer::setFeatures(FeatureParams params, const shared_ptr<ImageFilter>& filter, const shared_ptr<ImageFilter>& imageFilter) {
	featureParams = params;
	aspectRatio = params.windowAspectRatio();
	aspectRatioInv = 1.0 / aspectRatio;
	this->imageFilter = imageFilter;
	this->filter = filter;
	if (!imageFilter)
		featureExtractor = make_shared<AggregatedFeaturesExtractor>(filter,
				params.windowSizeInCells, params.cellSizeInPixels, params.octaveLayerCount);
	else
		featureExtractor = make_shared<AggregatedFeaturesExtractor>(imageFilter, filter,
				params.windowSizeInCells, params.cellSizeInPixels, params.octaveLayerCount);
}

void DetectorTrainer::train(vector<AnnotatedImage> images) {
	createEmptyClassifier();
	collectInitialTrainingExamples(images);
	trainClassifier();
	for (int round = 0; round < trainingParams.bootstrappingRounds; ++round) {
		collectHardTrainingExamples(images);
		retrainClassifier();
	}
}

void DetectorTrainer::createEmptyClassifier() {
	classifier = LibSvmClassifier::createBinarySvm(make_shared<LinearKernel>(),
			trainingParams.C, trainingParams.compensateImbalance, trainingParams.probabilistic);
	if (trainingParams.maxNegatives > 0)
		classifier->setNegativeExampleManagement(unique_ptr<ExampleManagement>(
				new HardNegativeExampleManagement(classifier, trainingParams.maxNegatives)));
}

void DetectorTrainer::collectInitialTrainingExamples(vector<AnnotatedImage> images) {
	if (printProgressInformation)
		std::cout << printPrefix << "collecting initial training examples" << std::endl;
	collectTrainingExamples(images, true);
}

void DetectorTrainer::collectHardTrainingExamples(vector<AnnotatedImage> images) {
	if (printProgressInformation)
		std::cout << printPrefix << "collecting additional hard negative training examples" << std::endl;
	createHardNegativesDetector();
	collectTrainingExamples(images, false);
}

void DetectorTrainer::createHardNegativesDetector() {
	classifier->getSvm()->setThreshold(trainingParams.negativeScoreThreshold);
	hardNegativesDetector = make_shared<AggregatedFeaturesDetector>(featureExtractor, classifier->getSvm(), noSuppression);
	classifier->getSvm()->setThreshold(0);
}

void DetectorTrainer::collectTrainingExamples(vector<AnnotatedImage> images, bool initial) {
	for (AnnotatedImage annotatedImage : images) {
		Annotations annotations = adjustSizes(annotatedImage.annotations);
		addTrainingExamples(annotatedImage.image, annotations, initial);
		if (trainingParams.mirrorTrainingData)
			addMirroredTrainingExamples(annotatedImage.image, annotations, initial);
	}
}

Annotations DetectorTrainer::adjustSizes(const Annotations& annotations) const {
	vector<Annotation> adjustedAnnotations;
	adjustedAnnotations.reserve(annotations.annotations.size());
	for (Annotation annotation : annotations.annotations)
		adjustedAnnotations.push_back(adjustSize(annotation));
	return Annotations{adjustedAnnotations};
}

Annotation DetectorTrainer::adjustSize(Annotation annotation) const {
	double cx = annotation.bounds.x + 0.5 * annotation.bounds.width;
	double cy = annotation.bounds.y + 0.5 * annotation.bounds.height;
	double width = featureParams.widthScaleFactor * annotation.bounds.width;
	double height = featureParams.heightScaleFactor * annotation.bounds.height;
	if (width < aspectRatio * height)
		width = aspectRatio * height;
	else if (width > aspectRatio * height)
		height = width * aspectRatioInv;
	double x = cx - 0.5 * width;
	double y = cy - 0.5 * height;
	return Annotation(cv::Rect_<double>(x, y, width, height), annotation.fuzzy);
}

void DetectorTrainer::addMirroredTrainingExamples(const Mat& image, const Annotations& annotations, bool initial) {
	Mat mirroredImage = flipHorizontally(image);
	Annotations mirroredAnnotations = flipHorizontally(annotations, image.cols);
	addTrainingExamples(mirroredImage, mirroredAnnotations, initial);
}

Mat DetectorTrainer::flipHorizontally(const Mat& image) {
	Mat flippedImage;
	cv::flip(image, flippedImage, 1);
	return flippedImage;
}

Annotations DetectorTrainer::flipHorizontally(const Annotations& annotations, int imageWidth) {
	vector<Annotation> flippedAnnotations;
	flippedAnnotations.reserve(annotations.annotations.size());
	for (Annotation annotation : annotations.annotations)
		flippedAnnotations.push_back(flipHorizontally(annotation, imageWidth));
	return Annotations{flippedAnnotations};
}

Annotation DetectorTrainer::flipHorizontally(Annotation annotation, int imageWidth) {
	annotation.bounds.x = imageWidth - (annotation.bounds.x + annotation.bounds.width);
	return annotation;
}

void DetectorTrainer::addTrainingExamples(const Mat& image, const Annotations& annotations, bool initial) {
	setImage(image);
	if (initial) {
		addPositiveExamples(annotations.positiveAnnotations());
		addRandomNegativeExamples(annotations.allAnnotations());
	} else {
		addHardNegativeExamples(annotations.allAnnotations());
	}
}

void DetectorTrainer::setImage(const Mat& image) {
	this->image = image;
	imageSize.width = image.cols;
	imageSize.height = image.rows;
	featureExtractor->update(image);
}

void DetectorTrainer::addPositiveExamples(const vector<Rect>& positiveBoxes) {
	for (const Rect& bounds : positiveBoxes) {
		shared_ptr<Patch> patch = featureExtractor->extract(bounds);
		if (patch)
			positiveTrainingExamples.push_back(patch->getData());
	}
}

void DetectorTrainer::addRandomNegativeExamples(const vector<Rect>& nonNegativeBoxes) {
	int addedCount = 0;
	while (addedCount < trainingParams.randomNegativesPerImage) {
		if (addNegativeIfNotOverlapping(createRandomBounds(), nonNegativeBoxes))
			++addedCount;
	}
}

Rect DetectorTrainer::createRandomBounds() const {
	typedef std::uniform_int_distribution<int> uniform_int;
	int minWidth = featureParams.windowSizeInPixels().width;
	int maxWidth = std::min(imageSize.width, static_cast<int>(imageSize.height * aspectRatio));
	int width = uniform_int{minWidth, maxWidth}(generator);
	int height = static_cast<int>(std::round(width * aspectRatioInv));
	int x = uniform_int{0, imageSize.width - width}(generator);
	int y = uniform_int{0, imageSize.height - height}(generator);
	return Rect(x, y, width, height);
}

void DetectorTrainer::addHardNegativeExamples(const vector<Rect>& nonNegativeBoxes) {
	vector<Rect> detections = hardNegativesDetector->detect(image);
	auto detection = detections.begin();
	int addedCount = 0;
	while (detection != detections.end() && addedCount < trainingParams.maxHardNegativesPerImage) {
		if (addNegativeIfNotOverlapping(*detection, nonNegativeBoxes))
			++addedCount;
		++detection;
	}
}

bool DetectorTrainer::addNegativeIfNotOverlapping(Rect candidate, const vector<Rect>& nonNegativeBoxes) {
	shared_ptr<Patch> patch = featureExtractor->extract(candidate);
	if (!patch || isOverlapping(patch->getBounds(), nonNegativeBoxes))
		return false;
	negativeTrainingExamples.push_back(patch->getData());
	return true;
}

bool DetectorTrainer::isOverlapping(Rect boxToTest, const vector<Rect>& otherBoxes) const {
	for (Rect otherBox : otherBoxes) {
		if (computeOverlap(boxToTest, otherBox) > trainingParams.overlapThreshold) {
			return true;
		}
	}
	return false;
}

double DetectorTrainer::computeOverlap(Rect a, Rect b) const {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}

void DetectorTrainer::trainClassifier() {
	trainClassifier(true);
}

void DetectorTrainer::retrainClassifier() {
	trainClassifier(false);
}

void DetectorTrainer::trainClassifier(bool initial) {
	if (printProgressInformation) {
		if (initial)
			std::cout << printPrefix << "training classifier (with " << positiveTrainingExamples.size() << " positives and " << negativeTrainingExamples.size() << " negatives)" << std::endl;
		else
			std::cout << printPrefix << "re-training classifier (found " << negativeTrainingExamples.size() << " potential new negatives)" << std::endl;
	}
	if (!classifier->retrain(positiveTrainingExamples, negativeTrainingExamples))
		throw runtime_error("DetectorTrainer: SVM is not usable after training");
	if (classifier->getSvm()->getSupportVectors().size() != 1) // should never happen because of linear kernel
		throw runtime_error("DetectorTrainer: the amount of support vectors has to be one (w)");
	positiveTrainingExamples.clear();
	negativeTrainingExamples.clear();
}
