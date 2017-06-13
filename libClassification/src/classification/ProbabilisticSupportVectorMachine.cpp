/*
 * ProbabilisticSupportVectorMachine.cpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticSupportVectorMachine.hpp"
#include <fstream>
#include <stdexcept>

using cv::Mat;
using std::pair;
using std::string;
using std::make_pair;
using std::shared_ptr;
using std::make_shared;

namespace classification {

ProbabilisticSupportVectorMachine::ProbabilisticSupportVectorMachine(shared_ptr<Kernel> kernel, double logisticA, double logisticB) :
		svm(make_shared<SupportVectorMachine>(kernel)), logisticA(logisticA), logisticB(logisticB) {}

ProbabilisticSupportVectorMachine::ProbabilisticSupportVectorMachine(shared_ptr<SupportVectorMachine> svm, double logisticA, double logisticB) :
		svm(svm), logisticA(logisticA), logisticB(logisticB) {}

bool ProbabilisticSupportVectorMachine::classify(const Mat& featureVector) const {
	return svm->classify(featureVector);
}

pair<bool, double> ProbabilisticSupportVectorMachine::getConfidence(const Mat& featureVector) const {
	return svm->getConfidence(featureVector);
}

pair<bool, double> ProbabilisticSupportVectorMachine::getProbability(const Mat& featureVector) const {
	return getProbability(svm->computeHyperplaneDistance(featureVector));
}

pair<bool, double> ProbabilisticSupportVectorMachine::getProbability(double hyperplaneDistance) const {
	double fABp = logisticA + logisticB * hyperplaneDistance;
	double probability = fABp >= 0 ? exp(-fABp) / (1.0 + exp(-fABp)) : 1.0 / (1.0 + exp(fABp));
	return make_pair(svm->classify(hyperplaneDistance), probability);
}

void ProbabilisticSupportVectorMachine::setLogisticParameters(double logisticA, double logisticB) {
	this->logisticA = logisticA;
	this->logisticB = logisticB;
}

void ProbabilisticSupportVectorMachine::setLogisticA(double logisticA) {
	this->logisticA = logisticA;
}

void ProbabilisticSupportVectorMachine::setLogisticB(double logisticB) {
	this->logisticB = logisticB;
}

void ProbabilisticSupportVectorMachine::store(std::ofstream& file) {
	svm->store(file);
	file << "Logistic " << logisticA << ' ' << logisticB << '\n';
}

std::shared_ptr<ProbabilisticSupportVectorMachine> ProbabilisticSupportVectorMachine::load(std::ifstream& file) {
	shared_ptr<SupportVectorMachine> svm = SupportVectorMachine::load(file);
	string tmp;
	double logisticA, logisticB;
	file >> tmp; // "Logistic"
	file >> logisticA;
	file >> logisticB;
	return make_shared<ProbabilisticSupportVectorMachine>(svm, logisticA, logisticB);
}

} /* namespace classification */
