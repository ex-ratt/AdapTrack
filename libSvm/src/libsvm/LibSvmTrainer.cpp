/*
 * LibSvmTrainer2.cpp
 *
 *  Created on: Jan 13, 2017
 *      Author: poschmann
 */

#include "libsvm/LibSvmTrainer.hpp"
#include <stdexcept>
#include <string>

using classification::Kernel;
using classification::ProbabilisticSvmClassifier;
using classification::SvmClassifier;
using cv::Mat;
using std::invalid_argument;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

namespace libsvm {

LibSvmTrainer::LibSvmTrainer(double c, bool compensateImbalance) : utils(), param() {
	param.reset(new struct svm_parameter);
	param->cache_size = 100;
	param->eps = 1e-4;
	param->C = c;
	param->svm_type = C_SVC;
	if (compensateImbalance) {
		param->nr_weight = 2;
		param->weight_label = (int*)malloc(param->nr_weight * sizeof(int));
		param->weight_label[0] = +1;
		param->weight_label[1] = -1;
		param->weight = (double*)malloc(param->nr_weight * sizeof(double));
		param->weight[0] = 1;
		param->weight[1] = 1;
	} else {
		param->nr_weight = 0;
		param->weight_label = nullptr;
		param->weight = nullptr;
	}
	param->shrinking = 0;
	param->gamma = 0;
	param->degree = 0;
}

void LibSvmTrainer::train(SvmClassifier& svm, const vector<Mat>& positives, const vector<Mat>& negatives) const {
	LibSvmData data = train(*svm.getKernel(), false, positives, negatives);
	setSvmParameters(svm, data.model.get());
}

void LibSvmTrainer::train(ProbabilisticSvmClassifier& svm, const vector<Mat>& positives, const vector<Mat>& negatives) const {
	LibSvmData data = train(*svm.getSvm()->getKernel(), true, positives, negatives);
	setSvmParameters(*svm.getSvm(), data.model.get());
	setLogisticParameters(svm, data.model.get());
}

LibSvmData LibSvmTrainer::train(const Kernel& kernel, bool probabilistic,
		const vector<Mat>& positives, const vector<Mat>& negatives) const {
	if (positives.empty() || negatives.empty())
		throw invalid_argument("LibSvmTrainer: there must be at least one positive and one negative training example");
	utils.setKernelParams(kernel, param.get());
	param->probability = probabilistic ? 1 : 0;
	LibSvmData data;
	data.positiveExamples = move(createNodes(positives));
	data.negativeExamples = move(createNodes(negatives));
	if (param->nr_weight == 2) { // compensate for imbalance in data
		double positiveCount = data.positiveExamples.size();
		double negativeCount = data.negativeExamples.size();
		param->weight[0] = negativeCount / positiveCount;
		param->weight[1] = positiveCount / negativeCount;
	}
	data.problem = move(createProblem(data.positiveExamples, data.negativeExamples));
	const char* message = svm_check_parameter(data.problem.get(), param.get());
	if (message != 0)
		throw invalid_argument(string("LibSvmTrainer: invalid SVM parameters: ") + message);
	data.model = unique_ptr<struct svm_model, ModelDeleter>(svm_train(data.problem.get(), param.get()));
	return data;
}

vector<unique_ptr<struct svm_node[], NodeDeleter>> LibSvmTrainer::createNodes(const vector<Mat>& examples) const {
	vector<unique_ptr<struct svm_node[], NodeDeleter>> nodes;
	nodes.reserve(examples.size());
	for (const Mat& example : examples)
		nodes.push_back(move(utils.createNode(example)));
	return move(nodes);
}

unique_ptr<struct svm_problem, ProblemDeleter> LibSvmTrainer::createProblem(
		const vector<unique_ptr<struct svm_node[], NodeDeleter>>& positiveExamples,
		const vector<unique_ptr<struct svm_node[], NodeDeleter>>& negativeExamples) const {
	unique_ptr<struct svm_problem, ProblemDeleter> problem(new struct svm_problem);
	problem->l = positiveExamples.size() + negativeExamples.size();
	problem->y = new double[problem->l];
	problem->x = new struct svm_node *[problem->l];
	size_t i = 0;
	for (auto& example : positiveExamples) {
		problem->y[i] = 1;
		problem->x[i] = example.get();
		++i;
	}
	for (auto& example : negativeExamples) {
		problem->y[i] = -1;
		problem->x[i] = example.get();
		++i;
	}
	return move(problem);
}

void LibSvmTrainer::setSvmParameters(SvmClassifier& svm, const struct svm_model* model) const {
	svm.setSvmParameters(utils.extractSupportVectors(model), utils.extractCoefficients(model), utils.extractBias(model));
}

void LibSvmTrainer::setLogisticParameters(ProbabilisticSvmClassifier& svm, const struct svm_model* model) const {
	// order of A and B in libSVM is reverse of order in ProbabilisticSvmClassifier
	// therefore ProbabilisticSvmClassifier.logisticA = libSVM.logisticB and vice versa
	svm.setLogisticParameters(utils.extractLogisticParamB(model), utils.extractLogisticParamA(model));
}

} /* namespace libsvm */
