/*
 *
 * PseudoProbabilisticSvmTrainer.hpp
 *  Created on: Jan 13, 2017
 *      Author: poschmann
 */

#ifndef CLASSIFICATION_PSEUDOPROBABILISTICSVMTRAINER_HPP_
#define CLASSIFICATION_PSEUDOPROBABILISTICSVMTRAINER_HPP_

#include "classification/ProbabilisticSupportVectorMachine.hpp"
#include "classification/SupportVectorMachine.hpp"
#include "classification/IncrementalClassifierTrainer.hpp"
#include <memory>

namespace classification {

/**
 * Probabilistic SVM trainer that assumes fixed mean positive and negative SVM outputs and computes
 * the logistic parameters only once on construction.
 */
class PseudoProbabilisticSvmTrainer : public IncrementalClassifierTrainer<ProbabilisticSupportVectorMachine> {
public:

	/**
	 * Constructs a new pseudo probabilistic SVM trainer with given logistic parameters.
	 *
	 * @param[in] svmTrainer SVM trainer.
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a * x + b)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a * x + b)).
	 */
	PseudoProbabilisticSvmTrainer(std::shared_ptr<IncrementalClassifierTrainer<SupportVectorMachine>> svmTrainer,
			double logisticA, double logisticB) :
			svmTrainer(svmTrainer), logisticA(logisticA), logisticB(logisticB) {}

	/**
	 * Constructs a new pseudo probabilistic SVM trainer, computing the logistic parameters on construction.
	 *
	 * @param[in] svmTrainer SVM trainer.
	 * @param[in] posProb Probability of the mean output of positive samples.
	 * @param[in] negProb Probability of the mean output of negative samples.
	 * @param[in] meanPosOutput Estimated mean SVM output of the positive samples.
	 * @param[in] meanNegOutput Estimated mean SVM output of the negative samples.
	 */
	PseudoProbabilisticSvmTrainer(std::shared_ptr<IncrementalClassifierTrainer<SupportVectorMachine>> svmTrainer,
			double posProb, double negProb, double meanPosOutput, double meanNegOutput) :
			svmTrainer(svmTrainer),
			logisticA((std::log((1 - negProb) / negProb) - std::log((1 - posProb) / posProb)) / (meanNegOutput - meanPosOutput)),
			logisticB(std::log((1 - posProb) / posProb) - logisticA * meanPosOutput) {}

	void train(ProbabilisticSupportVectorMachine& svm, const std::vector<cv::Mat>& positives, const std::vector<cv::Mat>& negatives) const {
		svmTrainer->train(*svm.getSvm(), positives, negatives);
		svm.setLogisticA(logisticA);
		svm.setLogisticB(logisticB);
	}

	void retrain(ProbabilisticSupportVectorMachine& svm, const std::vector<cv::Mat>& positives, const std::vector<cv::Mat>& negatives) const {
		svmTrainer->retrain(*svm.getSvm(), positives, negatives);
		svm.setLogisticA(logisticA);
		svm.setLogisticB(logisticB);
	}

private:

	std::shared_ptr<IncrementalClassifierTrainer<SupportVectorMachine>> svmTrainer; ///< SVM trainer.
	double logisticA; ///< Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a * x + b)).
	double logisticB; ///< Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a * x + b)).
};

} /* namespace classification */

#endif /* CLASSIFICATION_PSEUDOPROBABILISTICSVMTRAINER_HPP_ */
