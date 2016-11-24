/*
 * CombinedMeasurementModel.hpp
 *
 *  Created on: 02.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_COMBINEDMEASUREMENTMODEL_HPP_
#define TRACKING_FILTERING_COMBINEDMEASUREMENTMODEL_HPP_

#include "tracking/filtering/MeasurementModel.hpp"
#include <utility>
#include <vector>

namespace tracking {
namespace filtering {

/**
 * Measurement model that combines several measurement models into one, assuming independence.
 */
class CombinedMeasurementModel : public MeasurementModel {
public:

	CombinedMeasurementModel(std::shared_ptr<MeasurementModel> model1, std::shared_ptr<MeasurementModel> model2) :
			models({model1, model2}) {}

	CombinedMeasurementModel(std::shared_ptr<MeasurementModel> model1, std::shared_ptr<MeasurementModel> model2,
			std::shared_ptr<MeasurementModel> model3) :
					models({model1, model2, model3}) {}

	CombinedMeasurementModel(std::initializer_list<std::shared_ptr<MeasurementModel>> l) : models(l) {}

	CombinedMeasurementModel(std::vector<std::shared_ptr<MeasurementModel>> models) : models(models) {}

	void update(std::shared_ptr<imageprocessing::VersionedImage> image) override {
		for (std::shared_ptr<MeasurementModel> model : models)
			model->update(image);
	}

	double getLikelihood(const TargetState& state) const override {
		double likelihood = 1.0;
		for (std::shared_ptr<MeasurementModel> model : models)
			likelihood *= model->getLikelihood(state);
		return likelihood;
	}

private:

	std::vector<std::shared_ptr<MeasurementModel>> models; ///< Independent measurement models.

};

} /* namespace filtering */
} /* namespace tracking */

#endif /* TRACKING_FILTERING_COMBINEDMEASUREMENTMODEL_HPP_ */
