/*
 * CorrelatedCombinationModel.hpp
 *
 *  Created on: May 17, 2017
 *      Author: ex-ratt
 */

#ifndef TRACKING_FILTERING_CORRELATEDCOMBINATIONMODEL_HPP_
#define TRACKING_FILTERING_CORRELATEDCOMBINATIONMODEL_HPP_

namespace tracking {
namespace filtering {

/**
 * Measurement model that combines several measurement models with unknown correlations.
 */
class CorrelatedCombinationModel : public MeasurementModel {
public:

	CorrelatedCombinationModel(std::shared_ptr<MeasurementModel> model1, std::shared_ptr<MeasurementModel> model2) :
			models({model1, model2}), exponent(0.5) {}

	CorrelatedCombinationModel(std::shared_ptr<MeasurementModel> model1, std::shared_ptr<MeasurementModel> model2,
			std::shared_ptr<MeasurementModel> model3) :
					models({model1, model2, model3}), exponent(1.0 / 3.0) {}

	CorrelatedCombinationModel(std::initializer_list<std::shared_ptr<MeasurementModel>> l) :
			models(l), exponent(1.0 / models.size()) {}

	CorrelatedCombinationModel(std::vector<std::shared_ptr<MeasurementModel>> models) :
			models(models), exponent(1.0 / models.size()) {}

	void update(std::shared_ptr<imageprocessing::VersionedImage> image) override {
		for (std::shared_ptr<MeasurementModel> model : models)
			model->update(image);
	}

	double getLikelihood(const TargetState& state) const override {
		double likelihood = 1.0;
		for (std::shared_ptr<MeasurementModel> model : models)
			likelihood *= model->getLikelihood(state);
		return std::pow(likelihood, exponent);
	}

private:

	std::vector<std::shared_ptr<MeasurementModel>> models; ///< Potentially correlated measurement models.
	double exponent; ///< Exponent of the likelihood that prevents underestimating the resulting uncertainty.
};

} /* namespace filtering */
} /* namespace tracking */

#endif /* TRACKING_FILTERING_CORRELATEDCOMBINATIONMODEL_HPP_ */
