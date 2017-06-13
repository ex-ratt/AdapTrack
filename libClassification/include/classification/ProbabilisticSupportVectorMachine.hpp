/*
 * ProbabilisticSupportVectorMachine.hpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */

#ifndef CLASSIFICATION_PROBABILISTICSUPPORTVECTORMACHINE_HPP_
#define CLASSIFICATION_PROBABILISTICSUPPORTVECTORMACHINE_HPP_

#include "classification/ProbabilisticClassifier.hpp"
#include "classification/SupportVectorMachine.hpp"
#include <memory>

namespace classification {

/**
 * Logistic-function-based wrapper around a support vector machine that procudes pseudo-probabilistic output.
 *
 * The hyperplane distance of a feature vector will be transformed into a probability using a logistic function
 * p(x) = 1 / (1 + exp(a + b * x)) with x being the hyperplane distance and a and b being parameters.
 */
class ProbabilisticSupportVectorMachine : public ProbabilisticClassifier {
public:

	/**
	 * Constructs a new probabilistic support vector machine that creates the underlying SVM using the given kernel.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	explicit ProbabilisticSupportVectorMachine(std::shared_ptr<Kernel> kernel, double logisticA = 0.00556, double logisticB = -2.95);

	/**
	 * Constructs a new probabilistic support vector machine that is based on an already constructed SVM.
	 *
	 * @param[in] svm The actual support vector machine.
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	explicit ProbabilisticSupportVectorMachine(std::shared_ptr<SupportVectorMachine> svm, double logisticA = 0.00556, double logisticB = -2.95);

	bool classify(const cv::Mat& featureVector) const;

	std::pair<bool, double> getConfidence(const cv::Mat& featureVector) const;

	std::pair<bool, double> getProbability(const cv::Mat& featureVector) const;

	/**
	 * Computes the probability for being positive given the distance of a feature vector to the decision hyperplane.
	 *
	 * @param[in] hyperplaneDistance The distance of a feature vector to the decision hyperplane.
	 * @return A pair containing the binary classification result and a probability between zero and one for being positive.
	 */
	std::pair<bool, double> getProbability(double hyperplaneDistance) const;

	/**
	 * Changes the logistic parameters of this probabilistic support vector machine.
	 *
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	void setLogisticParameters(double logisticA, double logisticB);

	/**
	 * Changes logistic parameter a.
	 *
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	void setLogisticA(double logisticA);

	/**
	 * Changes logistic parameter b.
	 *
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	void setLogisticB(double logisticB);

	/**
	 * Creates a new probabilistic support vector machine from parameters (kernel, bias, coefficients, support vectors,
	 * logistic) given in a text file.
	 *
	 * @param[in] file The file input stream to load the parameters from.
	 * @return The newly created probabilistic support vector machine.
	 */
	static std::shared_ptr<ProbabilisticSupportVectorMachine> load(std::ifstream& file);

	/**
	 * Stores the logistic and SVM parameters (kernel, bias, coefficients, support vectors) into a text file.
	 *
	 * @param[in] file The file output stream to store the parameters into.
	 */
	void store(std::ofstream& file);

	/**
	 * @return The wrapped support vector machine.
	 */
	std::shared_ptr<SupportVectorMachine> getSvm() {
		return svm;
	}

	/**
	 * @return The wrapped support vector machine.
	 */
	const std::shared_ptr<SupportVectorMachine> getSvm() const {
		return svm;
	}

private:

	std::shared_ptr<SupportVectorMachine> svm; ///< The wrapped support vector machine.
	double logisticA; ///< Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	double logisticB; ///< Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
};

} /* namespace classification */
#endif /* CLASSIFICATION_PROBABILISTICSUPPORTVECTORMACHINE_HPP_ */

