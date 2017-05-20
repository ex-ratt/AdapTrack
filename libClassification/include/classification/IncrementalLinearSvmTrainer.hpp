/*
 * IncrementalLinearSvmTrainer.hpp
 *
 *  Created on: Jan 13, 2017
 *      Author: poschmann
 */

#ifndef CLASSIFICATION_INCREMENTALLINEARSVMTRAINER_HPP_
#define CLASSIFICATION_INCREMENTALLINEARSVMTRAINER_HPP_

#include "classification/IncrementalClassifierTrainer.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/SvmClassifier.hpp"
#include <memory>
#include <stdexcept>

namespace classification {

/**
 * Trainer of linear SVMs that approximately updates the SVM incrementally by training a new SVM
 * and computing a weighted average of the old and new weight and bias.
 */
class IncrementalLinearSvmTrainer : public IncrementalClassifierTrainer<SvmClassifier> {
public:

	/**
	 * Constructs a new incremental linear SVM trainer.
	 *
	 * @param[in] batchTrainer Batch SVM trainer.
	 * @param[in] learnRate Weight of the new SVM parameters (between zero and one).
	 */
	IncrementalLinearSvmTrainer(std::shared_ptr<ClassifierTrainer<SvmClassifier>> batchTrainer, double learnRate) :
			batchTrainer(batchTrainer), batchSvm(std::make_shared<LinearKernel>()), learnRate(learnRate) {
		if (learnRate < 0 || learnRate > 1)
			throw std::invalid_argument("IncrementalLinearSvmTrainer: the learn rate must be between zero (inclusive) and one (inclusive)");
	}

	void train(SvmClassifier& svm, const std::vector<cv::Mat>& positives, const std::vector<cv::Mat>& negatives) const override {
		if (!dynamic_cast<LinearKernel*>(svm.getKernel().get()))
			throw std::invalid_argument("IncrementalLinearSvmTrainer: the SVM must use a LinearKernel");
		batchTrainer->train(batchSvm, positives, negatives);
	  cv::Mat weight = batchSvm.getSupportVectors()[0];
	  double bias = batchSvm.getBias();
	  svm.setSvmParameters(std::vector<cv::Mat>{weight}, std::vector<float>{1}, bias);
	}

	void retrain(SvmClassifier& svm, const std::vector<cv::Mat>& positives, const std::vector<cv::Mat>& negatives) const override {
		if (!dynamic_cast<LinearKernel*>(svm.getKernel().get()))
			throw std::invalid_argument("IncrementalLinearSvmTrainer: the SVM must use a LinearKernel");
		if (svm.getSupportVectors().empty())
			throw std::invalid_argument("IncrementalLinearSvmTrainer: the SVM must have been trained at least once");
		batchTrainer->train(batchSvm, positives, negatives);
		cv::Mat weight = (1 - learnRate) * svm.getSupportVectors()[0] + learnRate * batchSvm.getSupportVectors()[0];
		double bias = (1 - learnRate) * svm.getBias() + learnRate * batchSvm.getBias();
		svm.setSvmParameters(std::vector<cv::Mat>{weight}, std::vector<float>{1}, bias);
	}

private:

	std::shared_ptr<ClassifierTrainer<SvmClassifier>> batchTrainer; ///< Batch SVM trainer.
	mutable SvmClassifier batchSvm; ///< SVM trained by the batch trainer.
	double learnRate; ///< Weight of the new SVM parameters (between zero and one).
};

} /* namespace classification */

#endif /* CLASSIFICATION_INCREMENTALLINEARSVMTRAINER_HPP_ */
