/*
 * DetectorTrainer.cpp
 *
 *  Created on: 21.10.2015
 *      Author: poschmann
 */

#include "DetectorTrainer.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/UnlimitedExampleManagement.hpp"
#include "imageprocessing/Patch.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

using classification::ClassifierTrainer;
using classification::LinearKernel;
using classification::ProbabilisticSupportVectorMachine;
using classification::SupportVectorMachine;
using classification::UnlimitedExampleManagement;
using cv::Mat;
using cv::Rect;
using cv::Size;
using detection::AggregatedFeaturesDetector;
using detection::NonMaximumSuppression;
using imageio::AnnotatedImage;
using imageio::Annotation;
using imageio::Annotations;
using imageprocessing::Patch;
using imageprocessing::extraction::AggregatedFeaturesExtractor;
using std::make_shared;
using std::make_unique;
using std::runtime_error;
using std::shared_ptr;
using std::string;
using std::vector;

shared_ptr<AggregatedFeaturesDetector> DetectorTrainer::getDetector(shared_ptr<NonMaximumSuppression> nms) const {
	return getDetector(nms, featureExtractor);
}

shared_ptr<AggregatedFeaturesDetector> DetectorTrainer::getDetector(shared_ptr<NonMaximumSuppression> nms,
		shared_ptr<AggregatedFeaturesExtractor> featureExtractor, float threshold) const {
	if (!svm)
		throw runtime_error("DetectorTrainer: must train the detector first");
	svm->setThreshold(threshold);
	shared_ptr<AggregatedFeaturesDetector> detector = make_shared<AggregatedFeaturesDetector>(featureExtractor, svm, nms);
	svm->setThreshold(0);
	return detector;
}

void DetectorTrainer::storeClassifier(const string& filename) const {
	std::ofstream stream(filename);
	if (probabilisticSvm)
		probabilisticSvm->store(stream);
	else
		svm->store(stream);
	stream.close();
}

Mat DetectorTrainer::getWeightVector() const {
	return svm->getSupportVectors().front();
}

void DetectorTrainer::setFeatureExtractor(shared_ptr<AggregatedFeaturesExtractor> featureExtractor) {
	this->featureExtractor = featureExtractor;
	Size patchSize = featureExtractor->getPatchSizeInCells();
	aspectRatio = static_cast<double>(patchSize.width) / static_cast<double>(patchSize.height);
	aspectRatioInv = 1.0 / aspectRatio;
}

void DetectorTrainer::setSvmTrainer(shared_ptr<ClassifierTrainer<SupportVectorMachine>> trainer) {
	svmTrainer = trainer;
	probabilisticSvmTrainer.reset();
}

void DetectorTrainer::setProbabilisticSvmTrainer(shared_ptr<ClassifierTrainer<ProbabilisticSupportVectorMachine>> trainer) {
	probabilisticSvmTrainer = trainer;
	svmTrainer.reset();
}

void DetectorTrainer::train(vector<AnnotatedImage> images) {
	if (!featureExtractor)
		throw runtime_error("DetectorTrainer: must set feature extractor first");
	if (!svmTrainer && !probabilisticSvmTrainer)
		throw runtime_error("DetectorTrainer: must set SVM trainer first");
	createEmptyClassifier();
	collectInitialTrainingExamples(images);
	trainClassifier();
	for (int round = 0; round < bootstrappingRounds; ++round) {
		collectHardTrainingExamples(images);
		retrainClassifier();
	}
}

void DetectorTrainer::createEmptyClassifier() {
	svm = make_shared<SupportVectorMachine>(make_shared<LinearKernel>());
	if (probabilisticSvmTrainer)
		probabilisticSvm = make_shared<ProbabilisticSupportVectorMachine>(svm);
	positives = make_unique<UnlimitedExampleManagement>();
	if (maxNegatives > 0)
		negatives = make_unique<HardNegativeExampleManagement>(svm, maxNegatives);
	else
		negatives = make_unique<UnlimitedExampleManagement>();
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
	svm->setThreshold(negativeScoreThreshold);
	hardNegativesDetector = make_shared<AggregatedFeaturesDetector>(featureExtractor, svm, noSuppression);
	svm->setThreshold(0);
}

void DetectorTrainer::collectTrainingExamples(vector<AnnotatedImage> images, bool initial) {
	for (AnnotatedImage annotatedImage : images) {
		Annotations annotations = adjustSizes(annotatedImage.annotations);
		addTrainingExamples(annotatedImage.image, annotations, initial);
		if (mirrorTrainingData)
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
	double width = annotation.bounds.width;
	double height = annotation.bounds.height;
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
	while (addedCount < randomNegativesPerImage) {
		if (addNegativeIfNotOverlapping(createRandomBounds(), nonNegativeBoxes))
			++addedCount;
	}
}

Rect DetectorTrainer::createRandomBounds() const {
	typedef std::uniform_int_distribution<int> uniform_int;
	int minWidth = featureExtractor->getPatchSizeInCells().width;
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
	while (detection != detections.end() && addedCount < maxHardNegativesPerImage) {
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
		if (computeOverlap(boxToTest, otherBox) > overlapThreshold) {
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
	if (probabilisticSvmTrainer)
		probabilisticSvmTrainer->train(*probabilisticSvm, positives->examples, negatives->examples);
	else
		svmTrainer->train(*svm, positives->examples, negatives->examples);
	newPositives.clear();
	newNegatives.clear();
}
