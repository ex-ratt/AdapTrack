/*
 * SupportVectorMachine.cpp
 *
 *  Created on: 17.02.2013
 *      Author: Patrik Huber
 */

#include "classification/HistogramIntersectionKernel.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/PolynomialKernel.hpp"
#include "classification/RbfKernel.hpp"
#include "classification/SupportVectorMachine.hpp"
#include <stdexcept>

using cv::Mat;
using std::pair;
using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::make_pair;
using std::runtime_error;

namespace classification {

SupportVectorMachine::SupportVectorMachine(shared_ptr<Kernel> kernel) :
		kernel(kernel), supportVectors(), coefficients(), bias(0), threshold(0) {}

bool SupportVectorMachine::classify(const Mat& featureVector) const {
	return classify(computeHyperplaneDistance(featureVector));
}

pair<bool, double> SupportVectorMachine::getConfidence(const Mat& featureVector) const {
	return getConfidence(computeHyperplaneDistance(featureVector));
}

bool SupportVectorMachine::classify(double hyperplaneDistance) const {
	return hyperplaneDistance >= threshold;
}

pair<bool, double> SupportVectorMachine::getConfidence(double hyperplaneDistance) const {
	if (classify(hyperplaneDistance))
		return make_pair(true, hyperplaneDistance);
	else
		return make_pair(false, -hyperplaneDistance);
}

double SupportVectorMachine::computeHyperplaneDistance(const Mat& featureVector) const {
	double distance = -bias;
	for (size_t i = 0; i < supportVectors.size(); ++i)
		distance += coefficients[i] * kernel->compute(featureVector, supportVectors[i]);
	return distance;
}

void SupportVectorMachine::store(std::ofstream& file) const {
	if (!file)
		throw runtime_error("SupportVectorMachine: Cannot write into stream");

	file << "Kernel ";
	if (dynamic_cast<LinearKernel*>(getKernel().get()))
		file << "Linear\n";
	else if (PolynomialKernel* kernel = dynamic_cast<PolynomialKernel*>(getKernel().get()))
		file << "Polynomial " << kernel->getDegree() << ' ' << kernel->getConstant() << ' ' << kernel->getAlpha() << '\n';
	else if (RbfKernel* kernel = dynamic_cast<RbfKernel*>(getKernel().get()))
		file << "RBF " << kernel->getGamma() << '\n';
	else if (dynamic_cast<HistogramIntersectionKernel*>(getKernel().get()))
		file << "HIK\n";
	else
		throw runtime_error("SupportVectorMachine: cannot write kernel parameters (unknown kernel type)");

	file << "Bias " << bias << '\n';
	file << "Coefficients " << coefficients.size() << '\n';
	for (float coefficient : coefficients)
		file << coefficient << '\n';
	const Mat& vector = supportVectors.front();
	int count = supportVectors.size();
	int rows = vector.rows;
	int cols = vector.cols;
	int channels = vector.channels();
	int depth = vector.depth();
	file << "SupportVectors " << count << ' ' << rows << ' ' << cols << ' ' << channels << ' ' << depth << '\n';
	switch (depth) {
		case CV_8U: storeSupportVectors<uchar>(file, supportVectors); break;
		case CV_32S: storeSupportVectors<int32_t>(file, supportVectors); break;
		case CV_32F: storeSupportVectors<float>(file, supportVectors); break;
		case CV_64F: storeSupportVectors<double>(file, supportVectors); break;
		default: throw runtime_error(
				"SupportVectorMachine: cannot store support vectors of depth other than CV_8U, CV_32S, CV_32F or CV_64F");
	}
}

shared_ptr<SupportVectorMachine> SupportVectorMachine::load(std::ifstream& file) {
	if (!file)
		throw runtime_error("SupportVectorMachine: Cannot read from stream");
	string tmp;

	file >> tmp; // "Kernel"
	shared_ptr<Kernel> kernel;
	string kernelType;
	file >> kernelType;
	if (kernelType == "Linear") {
		kernel.reset(new LinearKernel());
	} else if (kernelType == "Polynomial") {
		int degree;
		double constant, scale;
		file >> degree >> constant >> scale;
		kernel.reset(new PolynomialKernel(scale, constant, degree));
	} else if (kernelType == "RBF") {
		double gamma;
		file >> gamma;
		kernel.reset(new RbfKernel(gamma));
	} else if (kernelType == "HIK") {
		kernel.reset(new HistogramIntersectionKernel());
	} else {
		throw runtime_error("SupportVectorMachine: Invalid kernel type: " + kernelType);
	}
	shared_ptr<SupportVectorMachine> svm = make_shared<SupportVectorMachine>(kernel);

	file >> tmp; // "Bias"
	file >> svm->bias;

	size_t count;
	file >> tmp; // "Coefficients"
	file >> count;
	svm->coefficients.resize(count);
	for (size_t i = 0; i < count; ++i)
		file >> svm->coefficients[i];

	int rows, cols, channels, depth;
	file >> tmp; // "SupportVectors"
	file >> count; // should be the same as above
	file >> rows;
	file >> cols;
	file >> channels;
	file >> depth;
	switch (depth) {
		case CV_8U: loadSupportVectors<uchar>(file, count, rows, cols, channels, depth, svm->supportVectors); break;
		case CV_32S: loadSupportVectors<int32_t>(file, count, rows, cols, channels, depth, svm->supportVectors); break;
		case CV_32F: loadSupportVectors<float>(file, count, rows, cols, channels, depth, svm->supportVectors); break;
		case CV_64F: loadSupportVectors<double>(file, count, rows, cols, channels, depth, svm->supportVectors); break;
		default: throw runtime_error(
				"SupportVectorMachine: cannot load support vectors of depth other than CV_8U, CV_32S, CV_32F or CV_64F");
	}

	return svm;
}

} /* namespace classification */
