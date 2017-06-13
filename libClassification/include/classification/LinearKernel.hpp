/*
 * LinearKernel.hpp
 *
 *  Created on: 05.03.2013
 *      Author: poschmann
 */

#ifndef CLASSIFICATION_LINEARKERNEL_HPP_
#define CLASSIFICATION_LINEARKERNEL_HPP_

#include "classification/Kernel.hpp"
#include "classification/KernelVisitor.hpp"

namespace classification {

/**
 * A linear kernel of the form k(x, y) = x<sup>T</sup>y.
 */
class LinearKernel : public Kernel {
public:

	/**
	 * Constructs a new linear kernel.
	 */
	explicit LinearKernel() {}

	double compute(const cv::Mat& lhs, const cv::Mat& rhs) const {
		return lhs.dot(rhs);
	}

	void accept(KernelVisitor& visitor) const {
		visitor.visit(*this);
	}
};

} /* namespace classification */
#endif /* CLASSIFICATION_LINEARKERNEL_HPP_ */
