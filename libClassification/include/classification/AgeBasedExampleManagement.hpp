/*
 * AgeBasedExampleManagement.hpp
 *
 *  Created on: 26.11.2013
 *      Author: poschmann
 */

#ifndef CLASSIFICATION_AGEBASEDEXAMPLEMANAGEMENT_HPP_
#define CLASSIFICATION_AGEBASEDEXAMPLEMANAGEMENT_HPP_

#include "classification/ExampleManagement.hpp"

namespace classification {

/**
 * Example storage that, when reaching maximum size, replaces the oldest training examples with
 * new ones.
 */
class AgeBasedExampleManagement : public ExampleManagement {
public:

	/**
	 * Constructs a new age based example management.
	 *
	 * @param[in] capacity Maximum amount of stored training examples.
	 */
	explicit AgeBasedExampleManagement(size_t capacity);

	void add(const std::vector<cv::Mat>& newExamples);

private:

	size_t insertPosition; ///< The insertion index of new examples.
};

} /* namespace classification */
#endif /* CLASSIFICATION_AGEBASEDEXAMPLEMANAGEMENT_HPP_ */
