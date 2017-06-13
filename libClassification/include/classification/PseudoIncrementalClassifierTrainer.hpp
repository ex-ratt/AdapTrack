/*
 *
 * PseudoIncrementalClassifierTrainer.hpp
 *  Created on: Jan 13, 2017
 *      Author: poschmann
 */

#ifndef CLASSIFICATION_PSEUDOINCREMENTALCLASSIFIERTRAINER_HPP_
#define CLASSIFICATION_PSEUDOINCREMENTALCLASSIFIERTRAINER_HPP_

#include "classification/ClassifierTrainer.hpp"
#include "classification/ExampleManagement.hpp"
#include "classification/IncrementalClassifierTrainer.hpp"
#include "classification/UnlimitedExampleManagement.hpp"
#include <memory>

namespace classification {

/**
 * Incremental classifier trainer that carries out a batch training using the previous and new training examples.
 */
template<typename T>
class PseudoIncrementalClassifierTrainer : public IncrementalClassifierTrainer<T> {
public:

	PseudoIncrementalClassifierTrainer(std::shared_ptr<ClassifierTrainer<T>> batchTrainer) :
			batchTrainer(batchTrainer),
			positiveExamples(std::make_unique<UnlimitedExampleManagement>()),
			negativeExamples(std::make_unique<UnlimitedExampleManagement>()) {}

	void train(T& classifier, const std::vector<cv::Mat>& positives, const std::vector<cv::Mat>& negatives) const override {
		positiveExamples->clear();
		negativeExamples->clear();
		retrain(classifier, positives, negatives);
	}

	void retrain(T& classifier, const std::vector<cv::Mat>& positives, const std::vector<cv::Mat>& negatives) const override {
		if (!positives.empty() || !negatives.empty()) {
			positiveExamples->add(positives);
			negativeExamples->add(negatives);
			batchTrainer->train(classifier, positiveExamples->examples, negativeExamples->examples);
		}
	}

	/**
	 * @param[in] positiveExamples Storage of positive training examples.
	 */
	void setPositiveExampleManagement(std::unique_ptr<ExampleManagement> positiveExamples) {
		this->positiveExamples = move(positiveExamples);
	}

	/**
	 * @param[in] negativeExamples Storage of negative training examples.
	 */
	void setNegativeExampleManagement(std::unique_ptr<ExampleManagement> negativeExamples) {
		this->negativeExamples = move(negativeExamples);
	}

private:

	std::shared_ptr<ClassifierTrainer<T>> batchTrainer; ///< Batch trainer.
	std::unique_ptr<ExampleManagement> positiveExamples; ///< Storage of positive training examples.
	std::unique_ptr<ExampleManagement> negativeExamples; ///< Storage of negative training examples.
};

} /* namespace classification */

#endif /* CLASSIFICATION_PSEUDOINCREMENTALCLASSIFIERTRAINER_HPP_ */
