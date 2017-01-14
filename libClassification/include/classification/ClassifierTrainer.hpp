/*
 * ClassifierTrainer.hpp
 *
 *  Created on: Jan 13, 2017
 *      Author: poschmann
 */

#ifndef CLASSIFICATION_CLASSIFIERTRAINER_HPP_
#define CLASSIFICATION_CLASSIFIERTRAINER_HPP_

#include "opencv2/core/core.hpp"
#include <vector>

namespace classification {

/**
 * Trainer for classifiers.
 */
template<typename T>
class ClassifierTrainer {
public:

	virtual ~ClassifierTrainer() {}

	/**
	 * Trains the classifier with the given positive and negative training examples.
	 *
	 * @param[in] classifier Classifier to train.
	 * @param[in] positives Positive training examples.
	 * @param[in] negatives Negative training examples.
	 */
	virtual void train(T& classifier, const std::vector<cv::Mat>& positives, const std::vector<cv::Mat>& negatives) const = 0;
};

} /* namespace classification */

#endif /* CLASSIFICATION_CLASSIFIERTRAINER_HPP_ */
