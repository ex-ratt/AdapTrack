/*
 * SvmClassifier.cpp
 *
 *  Created on: 17.02.2013
 *      Author: Patrik Huber
 */

#include "classification/SvmClassifier.hpp"
#include "classification/HistogramIntersectionKernel.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/PolynomialKernel.hpp"
#include "classification/RbfKernel.hpp"
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

SvmClassifier::SvmClassifier(shared_ptr<Kernel> kernel) :
		VectorMachineClassifier(kernel), supportVectors(), coefficients() {}

bool SvmClassifier::classify(const Mat& featureVector) const {
	return classify(computeHyperplaneDistance(featureVector));
}

pair<bool, double> SvmClassifier::getConfidence(const Mat& featureVector) const {
	return getConfidence(computeHyperplaneDistance(featureVector));
}

bool SvmClassifier::classify(double hyperplaneDistance) const {
	return hyperplaneDistance >= threshold;
}

pair<bool, double> SvmClassifier::getConfidence(double hyperplaneDistance) const {
	if (classify(hyperplaneDistance))
		return make_pair(true, hyperplaneDistance);
	else
		return make_pair(false, -hyperplaneDistance);
}

double SvmClassifier::computeHyperplaneDistance(const Mat& featureVector) const {
	double distance = -bias;
	for (size_t i = 0; i < supportVectors.size(); ++i)
		distance += coefficients[i] * kernel->compute(featureVector, supportVectors[i]);
	return distance;
}

void SvmClassifier::setSvmParameters(vector<Mat> supportVectors, vector<float> coefficients, double bias) {
	this->supportVectors = supportVectors;
	this->coefficients = coefficients;
	this->bias = bias;
}

void SvmClassifier::store(std::ofstream& file) {
	if (!file)
		throw runtime_error("SvmClassifier: Cannot write into stream");

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
		throw runtime_error("SvmClassifier: cannot write kernel parameters (unknown kernel type)");

	file << "Bias " << getBias() << '\n';
	file << "Coefficients " << getCoefficients().size() << '\n';
	for (float coefficient : getCoefficients())
		file << coefficient << '\n';
	const Mat& vector = getSupportVectors().front();
	int count = getSupportVectors().size();
	int rows = vector.rows;
	int cols = vector.cols;
	int channels = vector.channels();
	int depth = vector.depth();
	file << "SupportVectors " << count << ' ' << rows << ' ' << cols << ' ' << channels << ' ' << depth << '\n';
	switch (depth) {
		case CV_8U: storeSupportVectors<uchar>(file, getSupportVectors()); break;
		case CV_32S: storeSupportVectors<int32_t>(file, getSupportVectors()); break;
		case CV_32F: storeSupportVectors<float>(file, getSupportVectors()); break;
		case CV_64F: storeSupportVectors<double>(file, getSupportVectors()); break;
		default: throw runtime_error(
				"SvmClassifier: cannot store support vectors of depth other than CV_8U, CV_32S, CV_32F or CV_64F");
	}
}

shared_ptr<SvmClassifier> SvmClassifier::load(std::ifstream& file) {
	if (!file)
		throw runtime_error("SvmClassifier: Cannot read from stream");
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
		throw runtime_error("SvmClassifier: Invalid kernel type: " + kernelType);
	}
	shared_ptr<SvmClassifier> svm = make_shared<SvmClassifier>(kernel);

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
				"SvmClassifier: cannot load support vectors of depth other than CV_8U, CV_32S, CV_32F or CV_64F");
	}

	return svm;
}

} /* namespace classification */
