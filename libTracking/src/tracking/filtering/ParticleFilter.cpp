/*
 * ParticleFilter.cpp
 *
 *  Created on: 01.11.2016
 *      Author: poschmann
 */

#include "tracking/filtering/ParticleFilter.hpp"
#include <stdexcept>

using cv::Rect;
using imageprocessing::VersionedImage;
using std::shared_ptr;
using std::vector;

namespace tracking {
namespace filtering {

ParticleFilter::ParticleFilter(shared_ptr<MotionModel> motionModel,
		shared_ptr<MeasurementModel> measurementModel, int count) :
				generator(std::random_device()()),
				standardUniform(0, 1),
				standardGaussian(0, 1),
				motionModel(motionModel),
				measurementModel(measurementModel),
				particles() {
	if (count < 1)
		throw std::invalid_argument("ParticleFilter: the number of particles must be greater than zero");
	particles.reserve(count);
}

void ParticleFilter::initialize(const shared_ptr<VersionedImage> image, const Rect& position,
		double positionDeviation, double velocityDeviation) {
	particles.clear();
	TargetState::setAspectRatio(position.width, position.height);
	int initialX = position.x + position.width / 2;
	int initialY = position.y + position.height / 2;
	int initialSize = position.width;
	double weight = 1.0 / particles.capacity();
	for (int i = 0; i < particles.capacity(); ++i) {
		int x = initialX + static_cast<int>(std::round(positionDeviation * initialSize * standardGaussian(generator)));
		int y = initialY + static_cast<int>(std::round(positionDeviation * initialSize * standardGaussian(generator)));
		int size = initialSize + static_cast<int>(std::round(positionDeviation * initialSize * standardGaussian(generator)));
		double velX = velocityDeviation * standardGaussian(generator);
		double velY = velocityDeviation * standardGaussian(generator);
		double velSize = velocityDeviation * standardGaussian(generator);
		particles.emplace_back(TargetState(x, y, size, velX, velY, velSize), weight);
	}
}

TargetState ParticleFilter::update(const shared_ptr<VersionedImage> image) {
	resampleParticles();
	moveParticles();
	weightParticles(image);
	return computeAverageState();
}

void ParticleFilter::resampleParticles() {
	int count = particles.size();
	vector<Particle> newParticles;
	newParticles.reserve(count);
	double weightStep = 1.0 / count;
	double weightPointer = weightStep * standardUniform(generator);
	double weightSum = 0;
	for (const Particle& particle : particles) {
		weightSum += particle.weight;
		while (weightSum > weightPointer) {
			newParticles.push_back(particle);
			weightPointer += weightStep;
		}
	}
	particles.swap(newParticles);
}

void ParticleFilter::moveParticles() {
	for (Particle& particle : particles)
		particle.state = motionModel->sample(particle.state);
}

void ParticleFilter::weightParticles(const shared_ptr<VersionedImage> image) {
	measurementModel->update(image);
	for (Particle& particle : particles)
		particle.weight *= measurementModel->getLikelihood(particle.state);
	normalizeParticleWeights();
}

void ParticleFilter::normalizeParticleWeights() {
	double weightSum = 0;
	for (const Particle& particle : particles)
		weightSum += particle.weight;
	if (!std::isfinite(weightSum))
		throw std::runtime_error("ParticleFilter: sum of particle weights is not finite: " + std::to_string(weightSum));
	if (weightSum > 0) {
		double normalizer = 1.0 / weightSum;
		for (Particle& particle : particles)
			particle.weight *= normalizer;
	} else { // weightSum == 0
		double weight = 1.0 / particles.size();
		for (Particle& particle : particles)
			particle.weight = weight;
	}
}

TargetState ParticleFilter::computeAverageState() {
	double x = 0;
	double y = 0;
	double s = 0;
	double velX = 0;
	double velY = 0;
	double velS = 0;
	for (const Particle& particle : particles) {
		x += particle.weight * particle.state.x;
		y += particle.weight * particle.state.y;
		s += particle.weight * particle.state.size;
		velX += particle.weight * particle.state.velX;
		velY += particle.weight * particle.state.velY;
		velS += particle.weight * particle.state.velSize;
	}
	return TargetState(x, y, s, velX, velY, velS);
}

} // namespace filtering
} // namespace tracking
