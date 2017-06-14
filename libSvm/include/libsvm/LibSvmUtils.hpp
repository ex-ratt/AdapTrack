/*
 * LibSvmUtils.hpp
 *
 *  Created on: 12.09.2013
 *      Author: poschmann
 */

#ifndef LIBSVM_LIBSVMUTILS_HPP_
#define LIBSVM_LIBSVMUTILS_HPP_

#include "svm.h"
#include "classification/Kernel.hpp"
#include "opencv2/core/core.hpp"
#include <unordered_map>
#include <memory>
#include <vector>

namespace libsvm {

/**
 * Deleter of libSVM nodes.
 */
class NodeDeleter {
public:
	void operator()(struct svm_node *node) const;
};

/**
 * Deleter of the libSVM parameter.
 */
class ParameterDeleter {
public:
	void operator()(struct svm_parameter *param) const;
};

/**
 * Deleter of the libSVM problem.
 */
class ProblemDeleter {
public:
	void operator()(struct svm_problem *problem) const;
};

/**
 * Deleter of the libSVM model.
 */
class ModelDeleter {
public:
	void operator()(struct svm_model *model) const;
};

/**
 * Utility class for libSVM with functions for creating nodes and computing SVM outputs. Usable via
 * composition or inheritance.
 */
class LibSvmUtils {
public:

	LibSvmUtils();

	virtual ~LibSvmUtils();

	/**
	 * Creates a new libSVM node from the given feature vector data.
	 *
	 * @param[in] vector The feature vector.
	 * @return The newly created libSVM node.
	 */
	std::unique_ptr<struct svm_node, NodeDeleter> createNode(const cv::Mat& vector) const;

	/**
	 * Creates a vector to the given libSVM node.
	 *
	 * @param[in] node The libSVM node.
	 * @return The feature vector.
	 */
	cv::Mat createVector(const struct svm_node& node) const;

	/**
	 * Sets the kernel parameters for libSVM training.
	 *
	 * @param[in] kernel The kernel.
	 * @param[in] params The libSVM parameters.
	 */
	void setKernelParams(const classification::Kernel& kernel, struct svm_parameter *params) const;

	/**
	 * Computes the SVM output given a libSVM node.
	 *
	 * @param[in] model The libSVM model.
	 * @param[in] x The libSVM node.
	 * @return The SVM output value.
	 */
	double computeSvmOutput(struct svm_model *model, const struct svm_node *x) const;

	/**
	 * Extracts the support vectors from a libSVM model.
	 *
	 * @param[in] model The libSVM model.
	 * @return The support vectors.
	 */
	std::vector<cv::Mat> extractSupportVectors(const struct svm_model *model) const;

	/**
	 * Extracts the coefficients from a libSVM model.
	 *
	 * @param[in] model The libSVM model.
	 * @return The coefficients.
	 */
	std::vector<float> extractCoefficients(const struct svm_model *model) const;

	/**
	 * Extracts the bias from a libSVM model.
	 *
	 * @param[in] model The libSVM model.
	 * @return The bias.
	 */
	double extractBias(const struct svm_model *model) const;

	/**
	 * Extracts parameter a from the logistic function that computes the probability
	 * p(x) = 1 / (1 + exp(a * x + b)) with x being the hyperplane distance.
	 *
	 * @param[in] model The libSVM model.
	 * @return The param a of the logistic function.
	 */
	double extractLogisticParamA(const struct svm_model *model) const;

	/**
	 * Extracts parameter b from the logistic function that computes the probability
	 * p(x) = 1 / (1 + exp(a * x + b)) with x being the hyperplane distance.
	 *
	 * @param[in] model The libSVM model.
	 * @return The param b of the logistic function.
	 */
	double extractLogisticParamB(const struct svm_model *model) const;

private:

	/**
	 * Fills a libSVM node with the data of a feature vector.
	 *
	 * @param[in,out] node The libSVM node.
	 * @param[in] vector The feature vector.
	 */
	template<class T>
	void fillNode(struct svm_node& node, const cv::Mat& vector) const;

	/**
	 * Fills a vector with the data of a libSVM node.
	 *
	 * @param[in,out] vector The vector.
	 * @param[in] node The libSVM node.
	 */
	template<class T>
	void fillMat(cv::Mat& vector, const struct svm_node& node) const;

	mutable int matRows;  ///< The row count of the support vector data.
	mutable int matCols;  ///< The column count of the support vector data.
	mutable int matType;  ///< The type of the support vector data.
	mutable int matDepth; ///< The depth of the support vector data.
};

} /* namespace libsvm */


#endif /* LIBSVM_LIBSVMUTILS_HPP_ */
