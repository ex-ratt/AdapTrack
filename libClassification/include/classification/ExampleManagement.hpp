/*
 * ExampleManagement.hpp
 *
 *  Created on: 25.11.2013
 *      Author: poschmann
 */

#ifndef CLASSIFICATION_EXAMPLEMANAGEMENT_HPP_
#define CLASSIFICATION_EXAMPLEMANAGEMENT_HPP_

#include "opencv2/core/core.hpp"
#include <vector>
#include <memory>

namespace classification {

/**
 * Stores and manages examples for training a classifier. Typically, the amount of training examples is budgeted,
 * meaning that there is a maximum amount of training examples that may be stored at a time.
 */
class ExampleManagement {
public:

	/**
	 * Constructs a new example management.
	 *
	 * @param[in] capacity Maximum amount of stored training examples.
	 */
	explicit ExampleManagement(size_t capacity) {
		examples.reserve(capacity);
	}

	virtual ~ExampleManagement() {}

	/**
	 * Adds new training examples, which may lead to the removal of some existing training examples.
	 *
	 * @param[in] newExamples Training examples to add.
	 */
	virtual void add(const std::vector<cv::Mat>& newExamples) = 0;

	/**
	 * Removes all training examples.
	 */
	virtual void clear()  {
		examples.clear();
	}

	std::vector<cv::Mat> examples; ///< Stored training examples.
};

} /* namespace classification */
#endif /* CLASSIFICATION_EXAMPLEMANAGEMENT_HPP_ */
