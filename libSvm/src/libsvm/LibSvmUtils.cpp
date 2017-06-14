/*
 * LibSvmUtils.cpp
 *
 *  Created on: 12.09.2013
 *      Author: poschmann
 */

#include "classification/Kernel.hpp"
#include "libsvm/LibSvmUtils.hpp"
#include "libsvm/LibSvmKernelParamSetter.hpp"
#include <stdexcept>

using classification::Kernel;
using cv::Mat;
using std::vector;
using std::unique_ptr;
using std::unordered_map;
using std::invalid_argument;

namespace libsvm {

LibSvmUtils::LibSvmUtils() :
		matRows(-1), matCols(-1), matType(CV_32FC1), matDepth(CV_32F) {}

LibSvmUtils::~LibSvmUtils() {}

unique_ptr<struct svm_node, NodeDeleter> LibSvmUtils::createNode(const Mat& vector) const {
	matRows = vector.rows;
	matCols = vector.cols;
	matType = vector.type();
	matDepth = vector.depth();
	if (matDepth != CV_8U && matDepth != CV_32F && matDepth != CV_64F)
		throw invalid_argument("LibSvmUtils: vector has to be of depth CV_8U, CV_32F, or CV_64F to create a node of");
	unique_ptr<struct svm_node, NodeDeleter> node(new struct svm_node);
	node->dim = vector.total() * vector.channels();
	node->values = new double[node->dim];
	if (matDepth == CV_8U)
		fillNode<uchar>(*node, vector);
	else if (matDepth == CV_32F)
		fillNode<float>(*node, vector);
	else if (matDepth == CV_64F)
		fillNode<double>(*node, vector);
	return move(node);
}

template<class T>
void LibSvmUtils::fillNode(struct svm_node& node, const Mat& vector) const {
	if (!vector.isContinuous())
		throw invalid_argument("LibSvmUtils: vector has to be continuous");
	const T* values = vector.ptr<T>();
	for (int i = 0; i < node.dim; ++i)
		node.values[i] = values[i];
}

Mat LibSvmUtils::createVector(const struct svm_node& node) const {
	Mat vector(matRows, matCols, matType);
	if (matDepth == CV_8U)
		fillMat<uchar>(vector, node);
	else if (matDepth == CV_32F)
		fillMat<float>(vector, node);
	else if (matDepth == CV_64F)
		fillMat<double>(vector, node);
	return vector;
}

template<class T>
void LibSvmUtils::fillMat(Mat& vector, const struct svm_node& node) const {
	T* values = vector.ptr<T>();
	for (int i = 0; i < node.dim; ++i)
		values[i] = static_cast<T>(node.values[i]);
}

void LibSvmUtils::setKernelParams(const Kernel& kernel, struct svm_parameter *params) const {
	LibSvmKernelParamSetter paramSetter(params);
	kernel.accept(paramSetter);
}

double LibSvmUtils::computeSvmOutput(struct svm_model *model, const struct svm_node *x) const {
	double* dec_values = new double[1];
	svm_predict_values(model, x, dec_values);
	double svmOutput = dec_values[0];
	delete[] dec_values;
	return svmOutput;
}

vector<Mat> LibSvmUtils::extractSupportVectors(const struct svm_model *model) const {
	if (model->param.kernel_type == LINEAR && (matDepth == CV_32F || matDepth == CV_64F)) {
		vector<Mat> supportVectors(1);
		supportVectors[0] = Mat::zeros(matRows, matCols, matType);
		if (matDepth == CV_32F) {
			float* values = supportVectors[0].ptr<float>();
			for (int i = 0; i < model->l; ++i) {
				double coeff = model->sv_coef[0][i];
				svm_node& node = model->SV[i];
				for (int i = 0; i < node.dim; ++i)
					values[i] += static_cast<float>(coeff * node.values[i]);
			}
		} else if (matDepth == CV_64F) {
			double* values = supportVectors[0].ptr<double>();
			for (int i = 0; i < model->l; ++i) {
				double coeff = model->sv_coef[0][i];
				svm_node& node = model->SV[i];
				for (int i = 0; i < node.dim; ++i)
					values[i] += coeff * node.values[i];
			}
		}
		return supportVectors;
	}
	vector<Mat> supportVectors;
	supportVectors.reserve(model->l);
	for (int i = 0; i < model->l; ++i)
		supportVectors.push_back(createVector(model->SV[i]));
	return supportVectors;
}

vector<float> LibSvmUtils::extractCoefficients(const struct svm_model *model) const {
	if (model->param.kernel_type == LINEAR && (matDepth == CV_32F || matDepth == CV_64F))
		return vector<float>{1};
	vector<float> coefficients;
	coefficients.reserve(model->l);
	for (int i = 0; i < model->l; ++i)
		coefficients.push_back(model->sv_coef[0][i]);
	return coefficients;
}

double LibSvmUtils::extractBias(const struct svm_model *model) const {
	return model->rho[0];
}

double LibSvmUtils::extractLogisticParamA(const struct svm_model *model) const {
	return model->probA[0];
}

double LibSvmUtils::extractLogisticParamB(const struct svm_model *model) const {
	return model->probB[0];
}

void NodeDeleter::operator()(struct svm_node *node) const {
	delete[] node->values;
	delete node;
}

void ParameterDeleter::operator()(struct svm_parameter *param) const {
	svm_destroy_param(param);
	delete param;
}

void ProblemDeleter::operator()(struct svm_problem *problem) const {
	delete[] problem->x;
	delete[] problem->y;
	delete problem;
}

void ModelDeleter::operator()(struct svm_model *model) const {
	svm_free_and_destroy_model(&model);
}

} /* namespace libsvm */
