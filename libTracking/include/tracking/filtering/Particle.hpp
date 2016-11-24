/*
 * Particle.hpp
 *
 *  Created on: 01.11.2016
 *      Author: poschmann
 */

#ifndef TRACKING_FILTERING_PARTICLE_HPP_
#define TRACKING_FILTERING_PARTICLE_HPP_

#include "opencv2/core/core.hpp"
#include "tracking/filtering/TargetState.hpp"

namespace tracking {
namespace filtering {

/**
 * Weighted particle that represents a possible state of a tracked target.
 */
class Particle {
public:

	/**
	 * Constructs a new default particle (zero state, weight of one).
	 */
	Particle() :
			state(), weight(1) {}

	/**
	 * Constructs a new particle.
	 *
	 * @param[in] state Target state.
	 * @param[in] weight Importance factor.
	 */
	Particle(const TargetState& state, double weight = 1) :
			state(state), weight(weight) {}

	/**
	 * Determines whether this particle is less than another particle using the weight. This particle is considered
	 * less than the other particle if the weight of this one is less than the weight of the other particle.
	 *
	 * @param[in] other Other particle.
	 * @return True if this particle comes before the other in a strict weak ordering, false otherwise.
	 */
	bool operator<(const Particle& other) const {
		return weight < other.weight;
	}

	/**
	 * Determines whether this particle is bigger than another particle using the weight. This particle is considered
	 * bigger than the other particle if the weight of this one is bigger than the weight of the other particle.
	 *
	 * @param[in] other Other particle.
	 * @return True if this particle comes before the other in a strict weak ordering, false otherwise.
	 */
	bool operator>(const Particle& other) const {
		return weight > other.weight;
	}

	/**
	 * Comparison function that compares particles by their weight in ascending order.
	 */
	class WeightComparisonAsc {
	public:
		bool operator()(const Particle& lhs, const Particle& rhs) {
			return lhs.weight < rhs.weight;
		}
	};

	/**
	 * Comparison function that compares particles by their weight in descending order.
	 */
	class WeightComparisonDesc {
	public:
		bool operator()(const Particle& lhs, const Particle& rhs) {
			return lhs.weight > rhs.weight;
		}
	};

	TargetState state; ///< Target state.
	double weight; ///< Importance factor.
};

} // namespace filtering
} // namespace tracking

#endif /* TRACKING_FILTERING_PARTICLE_HPP_ */
