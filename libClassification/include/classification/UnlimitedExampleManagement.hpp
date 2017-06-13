/*
 * UnlimitedExampleManagement.hpp
 *
 *  Created on: 26.11.2013
 *      Author: poschmann
 */

#ifndef CLASSIFICATION_UNLIMITEDEXAMPLEMANAGEMENT_HPP_
#define CLASSIFICATION_UNLIMITEDEXAMPLEMANAGEMENT_HPP_

#include "classification/ExampleManagement.hpp"

namespace classification {

/**
 * Example storage that never replaces existing training examples (unless cleared).
 */
class UnlimitedExampleManagement : public ExampleManagement {
public:

	/**
	 * Constructs a new unlimited example management.
	 */
	UnlimitedExampleManagement() : ExampleManagement(10) {}

	void add(const std::vector<cv::Mat>& newExamples) {
		for (const cv::Mat& example : newExamples)
			examples.push_back(example);
	}
};

} /* namespace classification */
#endif /* CLASSIFICATION_UNLIMITEDEXAMPLEMANAGEMENT_HPP_ */
