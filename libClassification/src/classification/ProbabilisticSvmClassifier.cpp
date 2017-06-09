/*
 * ProbabilisticSvmClassifier.cpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include <fstream>
#include <stdexcept>

using cv::Mat;
using std::pair;
using std::string;
using std::make_pair;
using std::shared_ptr;
using std::make_shared;

namespace classification {

ProbabilisticSvmClassifier::ProbabilisticSvmClassifier(shared_ptr<Kernel> kernel, double logisticA, double logisticB) :
		svm(make_shared<SvmClassifier>(kernel)), logisticA(logisticA), logisticB(logisticB) {}

ProbabilisticSvmClassifier::ProbabilisticSvmClassifier(shared_ptr<SvmClassifier> svm, double logisticA, double logisticB) :
		svm(svm), logisticA(logisticA), logisticB(logisticB) {}

bool ProbabilisticSvmClassifier::classify(const Mat& featureVector) const {
	return svm->classify(featureVector);
}

pair<bool, double> ProbabilisticSvmClassifier::getConfidence(const Mat& featureVector) const {
	return svm->getConfidence(featureVector);
}

pair<bool, double> ProbabilisticSvmClassifier::getProbability(const Mat& featureVector) const {
	return getProbability(svm->computeHyperplaneDistance(featureVector));
}

pair<bool, double> ProbabilisticSvmClassifier::getProbability(double hyperplaneDistance) const {
	double fABp = logisticA + logisticB * hyperplaneDistance;
	double probability = fABp >= 0 ? exp(-fABp) / (1.0 + exp(-fABp)) : 1.0 / (1.0 + exp(fABp));
	return make_pair(svm->classify(hyperplaneDistance), probability);
}

void ProbabilisticSvmClassifier::setLogisticParameters(double logisticA, double logisticB) {
	this->logisticA = logisticA;
	this->logisticB = logisticB;
}

void ProbabilisticSvmClassifier::setLogisticA(double logisticA) {
	this->logisticA = logisticA;
}

void ProbabilisticSvmClassifier::setLogisticB(double logisticB) {
	this->logisticB = logisticB;
}

void ProbabilisticSvmClassifier::store(std::ofstream& file) {
	svm->store(file);
	file << "Logistic " << logisticA << ' ' << logisticB << '\n';
}

std::shared_ptr<ProbabilisticSvmClassifier> ProbabilisticSvmClassifier::load(std::ifstream& file) {
	shared_ptr<SvmClassifier> svm = SvmClassifier::load(file);
	string tmp;
	double logisticA, logisticB;
	file >> tmp; // "Logistic"
	file >> logisticA;
	file >> logisticB;
	return make_shared<ProbabilisticSvmClassifier>(svm, logisticA, logisticB);
}

} /* namespace classification */
