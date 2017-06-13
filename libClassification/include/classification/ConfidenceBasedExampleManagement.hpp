/*
 * ConfidenceBasedExampleManagement.hpp
 *
 *  Created on: 25.11.2013
 *      Author: poschmann
 */

#ifndef CLASSIFICATION_CONFIDENCEBASEDEXAMPLEMANAGEMENT_HPP_
#define CLASSIFICATION_CONFIDENCEBASEDEXAMPLEMANAGEMENT_HPP_

#include "classification/BinaryClassifier.hpp"
#include "classification/ExampleManagement.hpp"

namespace classification {

/**
 * Example storage that, when reaching maximum size, replaces training examples that have the highest
 * confidence when evaluated by the classifier. The first training examples will not be replaced
 * (keeps the first example by default).
 */
class ConfidenceBasedExampleManagement : public ExampleManagement {
public:

	/**
	 * Constructs a new confidence based example management.
	 *
	 * @param[in] classifier Classifier for computing the confidences of the training examples.
	 * @param[in] positive Flag that indicates whether this set contains positive training examples.
	 * @param[in] capacity Maximum amount of stored training examples.
	 */
	ConfidenceBasedExampleManagement(const std::shared_ptr<BinaryClassifier>& classifier, bool positive, size_t capacity);

	virtual ~ConfidenceBasedExampleManagement() {}

	/**
	 * Changes the number of initial training examples that should never be replaced.
	 *
	 * @param[in] keep The new number of initial training examples to never replace.
	 */
	void setFirstExamplesToKeep(size_t keep);

	void add(const std::vector<cv::Mat>& newExamples);

private:

	const std::shared_ptr<BinaryClassifier> classifier; ///< Classifier for computing the confidences of the training examples.
	bool positive; ///< Flag that indicates whether this set contains positive training examples.
	size_t keep;
};

} /* namespace classification */
#endif /* CLASSIFICATION_CONFIDENCEBASEDEXAMPLEMANAGEMENT_HPP_ */
