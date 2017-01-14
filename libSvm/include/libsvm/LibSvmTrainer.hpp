/*
 * LibSvmTrainer.hpp
 *
 *  Created on: Jan 13, 2017
 *      Author: poschmann
 */

#ifndef LIBSVM_LIBSVMTRAINER_HPP_
#define LIBSVM_LIBSVMTRAINER_HPP_

#include "svm.h"
#include "classification/ClassifierTrainer.hpp"
#include "classification/Kernel.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include "libsvm/LibSvmUtils.hpp"
#include <memory>

namespace libsvm {

struct LibSvmData {
	std::vector<std::unique_ptr<struct svm_node[], NodeDeleter>> positiveExamples;
	std::vector<std::unique_ptr<struct svm_node[], NodeDeleter>> negativeExamples;
	std::unique_ptr<struct svm_problem, ProblemDeleter> problem;
	std::unique_ptr<struct svm_model, ModelDeleter> model;
};

/**
 * SVM trainer based on libSVM.
 */
class LibSvmTrainer :
		public classification::ClassifierTrainer<classification::SvmClassifier>,
		public classification::ClassifierTrainer<classification::ProbabilisticSvmClassifier> {
public:

	/**
	 * Constructs a new libSVM trainer.
	 *
	 * @param[in] c Soft margin parameter.
	 * @param[in] compensateImbalance Flag that indicates whether to adjust class weights to compensate for unbalanced data.
	 */
	LibSvmTrainer(double c, bool compensateImbalance);

	void train(classification::SvmClassifier& svm,
			const std::vector<cv::Mat>& positives, const std::vector<cv::Mat>& negatives) const override;

	void train(classification::ProbabilisticSvmClassifier& svm,
			const std::vector<cv::Mat>& positives, const std::vector<cv::Mat>& negatives) const override;

private:

	/**
	 * Trains a new SVM model.
	 *
	 * @param[in] kernel SVM kernel.
	 * @param[in] probabilistic Flag that indicates whether to compute logistic parameters for probabilistic output.
	 * @param[in] positives Positive training examples.
	 * @param[in] negatives Negative training examples.
	 * @return Data of the training.
	 */
	LibSvmData train(const classification::Kernel& kernel, bool probabilistic,
			const std::vector<cv::Mat>& positives, const std::vector<cv::Mat>& negatives) const;

	/**
	 * Creates libSVM nodes from training examples.
	 *
	 * @param[in] examples Training examples.
	 * @return Vector of libSVM nodes.
	 */
	std::vector<std::unique_ptr<struct svm_node[], NodeDeleter>> createNodes(const std::vector<cv::Mat>& examples) const;

	/**
	 * Creates the libSVM problem containing the training data.
	 *
	 * @param[in] positiveExamples Positive training examples.
	 * @param[in] negativeExamples Negative training examples.
	 * @return The libSVM problem.
	 */
	std::unique_ptr<struct svm_problem, ProblemDeleter> createProblem(
			const std::vector<std::unique_ptr<struct svm_node[], NodeDeleter>>& positiveExamples,
			const std::vector<std::unique_ptr<struct svm_node[], NodeDeleter>>& negativeExamples) const;

	/**
	 * Sets parameters of the SVM according to the model.
	 *
	 * @param[in] svm SVM whose parameters to change.
	 * @param[in] model SVM model to take the parameter values from.
	 */
	void setSvmParameters(classification::SvmClassifier& svm, const struct svm_model* model) const;

	/**
	 * Sets parameters of the probabilistic SVM's logistic function according to the model.
	 *
	 * @param[in] svm Probabilistic SVM whose logistic parameters to change.
	 * @param[in] model SVM model to take the parameter values from.
	 */
	void setLogisticParameters(classification::ProbabilisticSvmClassifier& svm, const struct svm_model* model) const;

	LibSvmUtils utils; ///< Utils for using libSVM.
	std::unique_ptr<struct svm_parameter, ParameterDeleter> param; ///< Parameters of libSVM.
};

} /* namespace libsvm */

#endif /* LIBSVM_LIBSVMTRAINER_HPP_ */
