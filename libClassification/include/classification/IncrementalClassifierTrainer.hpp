/*
 * IncrementalClassifierTrainer.hpp
 *
 *  Created on: Jan 13, 2017
 *      Author: poschmann
 */

#ifndef CLASSIFICATION_INCREMENTALCLASSIFIERTRAINER_HPP_
#define CLASSIFICATION_INCREMENTALCLASSIFIERTRAINER_HPP_

#include "classification/ClassifierTrainer.hpp"

namespace classification {

/**
 * Incremental trainer for classifiers.
 */
template<typename T>
class IncrementalClassifierTrainer : public ClassifierTrainer<T> {
public:

	virtual ~IncrementalClassifierTrainer() {}

	/**
	 * Incrementally retrains the classifier with the new positive and negative training examples.
	 *
	 * @param[in] classifier Classifier to update.
	 * @param[in] positives New positive training examples.
	 * @param[in] negatives New negative training examples.
	 */
	virtual void retrain(T& classifier, const std::vector<cv::Mat>& positives, const std::vector<cv::Mat>& negatives) const = 0;
};

} /* namespace classification */

#endif /* CLASSIFICATION_INCREMENTALCLASSIFIERTRAINER_HPP_ */
