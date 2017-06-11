/*
 * DetectorTrainer.cpp
 *
 *  Created on: 21.10.2015
 *      Author: poschmann
 */

#include "DetectorTrainer.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/UnlimitedExampleManagement.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/Patch.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

using classification::LinearKernel;
using classification::ProbabilisticSvmClassifier;
using classification::SvmClassifier;
using classification::UnlimitedExampleManagement;
using cv::Mat;
using cv::Rect;
using detection::AggregatedFeaturesDetector;
using detection::NonMaximumSuppression;
using imageio::AnnotatedImage;
using imageio::Annotation;
using imageio::Annotations;
using imageprocessing::Patch;
using imageprocessing::extraction::AggregatedFeaturesExtractor;
using imageprocessing::filtering::ImageFilter;
using libsvm::LibSvmTrainer;
using std::make_shared;
using std::make_unique;
using std::runtime_error;
using std::shared_ptr;
using std::string;
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
	if (!svm)
		throw runtime_error("DetectorTrainer: must train the detector first");
	svm->setThreshold(threshold);
	shared_ptr<AggregatedFeaturesDetector> detector;
	if (!imageFilter)
		detector = make_shared<AggregatedFeaturesDetector>(filter, featureParams.cellSizeInPixels,
				featureParams.windowSizeInCells, octaveLayerCount, svm, nms,
				featureParams.widthScaleFactorInv(), featureParams.heightScaleFactorInv());
	else
		detector = make_shared<AggregatedFeaturesDetector>(imageFilter, filter, featureParams.cellSizeInPixels,
				featureParams.windowSizeInCells, octaveLayerCount, svm, nms,
				featureParams.widthScaleFactorInv(), featureParams.heightScaleFactorInv());
	svm->setThreshold(0);
	return detector;
}

void DetectorTrainer::storeClassifier(const string& filename) const {
	std::ofstream stream(filename);
	if (trainingParams.probabilistic)
		probabilisticSvm->store(stream);
	else
		svm->store(stream);
	stream.close();
}

Mat DetectorTrainer::getWeightVector() const {
	return svm->getSupportVectors().front();
}

void DetectorTrainer::setTrainingParameters(TrainingParams params) {
	trainingParams = params;
	trainer = make_shared<LibSvmTrainer>(trainingParams.C, trainingParams.compensateImbalance);
	positives = make_unique<UnlimitedExampleManagement>();
	if (trainingParams.maxNegatives > 0)
		negatives = make_unique<HardNegativeExampleManagement>(svm, trainingParams.maxNegatives);
	else
		negatives = make_unique<UnlimitedExampleManagement>();
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
	if (!featureExtractor)
		throw runtime_error("DetectorTrainer: must set feature parameters first");
	if (!trainer)
		throw runtime_error("DetectorTrainer: must set training parameters first");
	createEmptyClassifier();
	collectInitialTrainingExamples(images);
	trainClassifier();
	for (int round = 0; round < trainingParams.bootstrappingRounds; ++round) {
		collectHardTrainingExamples(images);
		retrainClassifier();
	}
}

void DetectorTrainer::createEmptyClassifier() {
	svm = make_shared<SvmClassifier>(make_shared<LinearKernel>());
	probabilisticSvm = make_shared<ProbabilisticSvmClassifier>(svm);
	positives->clear();
	negatives->clear();
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
	svm->setThreshold(trainingParams.negativeScoreThreshold);
	hardNegativesDetector = make_shared<AggregatedFeaturesDetector>(featureExtractor, svm, noSuppression);
	svm->setThreshold(0);
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
			newPositives.push_back(patch->getData());
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
	newNegatives.push_back(patch->getData());
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
	if (printProgressInformation)
		std::cout << printPrefix << "training SVM (with " << newPositives.size() << " positives and " << newNegatives.size() << " negatives)" << std::endl;
	trainSvm();
}

void DetectorTrainer::retrainClassifier() {
	if (printProgressInformation)
		std::cout << printPrefix << "re-training SVM (found " << newNegatives.size() << " potential new negatives)" << std::endl;
	trainSvm();
}

void DetectorTrainer::trainSvm() {
	positives->add(newPositives);
	negatives->add(newNegatives);
	if (trainingParams.probabilistic)
		trainer->train(*probabilisticSvm, positives->getAll(), negatives->getAll());
	else
		trainer->train(*svm, positives->getAll(), negatives->getAll());
	newPositives.clear();
	newNegatives.clear();
}
