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
		throw std::invalid_argument("the number of particles must be greater than zero");
	particles.reserve(count);
}

void ParticleFilter::initialize(const shared_ptr<VersionedImage> image, const Rect& position,
		double positionDeviation, double velocityDeviation) {
	particles.clear();
	Particle::setAspectRatio(position.width, position.height);
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
		particles.emplace_back(x, y, size, velX, velY, velSize, weight);
	}
}

Rect ParticleFilter::update(const shared_ptr<VersionedImage> image) {
	resampleParticles();
	moveParticles();
	weightParticles(image);
	return computeAverageBounds();
}

void ParticleFilter::resampleParticles() {
	int count = particles.size();
	vector<Particle> newParticles;
	newParticles.reserve(count);
	double weightStep = 1.0 / count;
	double weightPointer = weightStep * standardUniform(generator);
	double weightSum = 0;
	for (const Particle& particle : particles) {
		weightSum += particle.getWeight();
		while (weightSum > weightPointer) {
			newParticles.push_back(particle);
			weightPointer += weightStep;
		}
	}
	particles.swap(newParticles);
}

void ParticleFilter::moveParticles() {
	for (Particle& particle : particles)
		motionModel->sample(particle);
}

void ParticleFilter::weightParticles(const shared_ptr<VersionedImage> image) {
	measurementModel->update(image);
	for (Particle& particle : particles)
		measurementModel->evaluate(particle);
	normalizeParticleWeights();
}

void ParticleFilter::normalizeParticleWeights() {
	double weightSum = 0;
	for (const Particle& particle : particles)
		weightSum += particle.getWeight();
	if (!std::isfinite(weightSum))
		throw std::runtime_error("sum of particle weights is not finite: " + std::to_string(weightSum));
	if (weightSum > 0) {
		double normalizer = 1.0 / weightSum;
		for (Particle& particle : particles)
			particle.setWeight(normalizer * particle.getWeight());
	} else { // weightSum == 0
		double weight = 1.0 / particles.size();
		for (Particle& particle : particles)
			particle.setWeight(weight);
	}
}

Rect ParticleFilter::computeAverageBounds() {
	double x = 0;
	double y = 0;
	double s = 0;
	for (const Particle& particle : particles) {
		x += particle.getWeight() * particle.getX();
		y += particle.getWeight() * particle.getY();
		s += particle.getWeight() * particle.getSize();
	}
	return Particle(x, y, s).getBounds();
}

} // namespace filtering

} // namespace tracking
