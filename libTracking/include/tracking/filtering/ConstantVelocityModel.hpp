/*
 * ConstantVelocityModel.hpp
 *
 *  Created on: 01.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_CONSTANTVELOCITYMODEL_HPP_
#define TRACKING_FILTERING_CONSTANTVELOCITYMODEL_HPP_

#include "opencv2/core/core.hpp"
#include "tracking/filtering/MotionModel.hpp"
#include <random>
#include <stdexcept>

namespace tracking {
namespace filtering {

/**
 * Motion model that assumes a constant velocity with Gaussian process noise.
 */
class ConstantVelocityModel : public MotionModel {
public:

	/**
	 * Constructs a new constant velocity model with the same standard deviation for position and size velocity.
	 *
	 * @param[in] deviation Standard deviation of the position and size velocity noise.
	 */
	explicit ConstantVelocityModel(double deviation) : ConstantVelocityModel(deviation, deviation) {}

	/**
	 * Constructs a new constant velocity model with different standard deviations for position and size velocity.
	 *
	 * @param[in] positionDeviation Standard deviation of the position velocity noise.
	 * @param[in] sizeDeviation Standard deviation of the size velocity noise.
	 */
	explicit ConstantVelocityModel(double positionDeviation, double sizeDeviation) :
			generator(std::random_device()()),
			standardGaussian(0, 1),
			positionDeviation(positionDeviation),
			sizeDeviation(sizeDeviation),
			L(cv::Mat::zeros(6, 6, CV_64FC1)) {

		if (positionDeviation <= 0.0 || sizeDeviation <= 0.0)
			throw new std::invalid_argument("ConstantVelocityModel: the standard deviations must be bigger than zero");
		double positionVelocityVariancePerFrame = positionDeviation * positionDeviation;
		double sizeVelocityVariancePerFrame = sizeDeviation * sizeDeviation;

		L.at<double>(0, 0) = positionVelocityVariancePerFrame / 3;
		L.at<double>(1, 1) = positionVelocityVariancePerFrame / 3;
		L.at<double>(2, 2) = sizeVelocityVariancePerFrame / 3;
		L.at<double>(3, 3) = positionVelocityVariancePerFrame;
		L.at<double>(4, 4) = positionVelocityVariancePerFrame;
		L.at<double>(5, 5) = sizeVelocityVariancePerFrame;

		L.at<double>(0, 3) = positionVelocityVariancePerFrame / 2;
		L.at<double>(1, 4) = positionVelocityVariancePerFrame / 2;
		L.at<double>(2, 5) = sizeVelocityVariancePerFrame / 2;
		L.at<double>(3, 0) = positionVelocityVariancePerFrame / 2;
		L.at<double>(4, 1) = positionVelocityVariancePerFrame / 2;
		L.at<double>(5, 2) = sizeVelocityVariancePerFrame / 2;

		if (!cv::Cholesky(L.ptr<double>(), L.step, L.cols, nullptr, 0, 0))
			throw std::runtime_error("ConstantVelocityModel: Cholesky decomposition did not work");

		for (int i = 0; i < L.rows; ++i) {
			L.at<double>(i, i) = 1.0 / L.at<double>(i, i);
			for (int j = i + 1; j < L.cols; ++j)
				L.at<double>(i, j) = 0;
		}
	}

	TargetState sample(const TargetState& state) const override {
		cv::Mat standardRandomValues(6, 1, CV_64FC1);
		for (int i = 0; i < standardRandomValues.rows; ++i)
			standardRandomValues.at<double>(i) = standardGaussian(generator);
		cv::Mat correlatedRandomValues = L * standardRandomValues;
		int x = static_cast<int>(std::round(state.x + (state.velX + correlatedRandomValues.at<double>(0)) * state.size));
		int y = static_cast<int>(std::round(state.y + (state.velY + correlatedRandomValues.at<double>(1)) * state.size));
		int size = static_cast<int>(std::round(state.size + (state.velSize + correlatedRandomValues.at<double>(2)) * state.size));
		double velX = state.velX + correlatedRandomValues.at<double>(3);
		double velY = state.velY + correlatedRandomValues.at<double>(4);
		double velSize = state.velSize + correlatedRandomValues.at<double>(5);
		return TargetState(x, y, size, velX, velY, velSize);
	}

private:

	mutable std::default_random_engine generator; ///< Random number generator.
	mutable std::normal_distribution<> standardGaussian; ///< Normal distribution with zero mean and unit variance.
	double positionDeviation; ///< Standard deviation of the position velocity noise.
	double sizeDeviation; ///< Standard deviation of the size velocity noise.
	cv::Mat L; ///< Lower triangle matrix of the Cholesky decomposition of the covariance matrix.
};

} // namespace filtering
} // namespace tracking

#endif /* TRACKING_FILTERING_CONSTANTVELOCITYMODEL_HPP_ */
