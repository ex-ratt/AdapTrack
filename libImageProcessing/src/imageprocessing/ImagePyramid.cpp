/*
 * ImagePyramid.cpp
 *
 *  Created on: 15.02.2013
 *      Author: huber & poschmann
 */

#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/filtering/ChainedFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdexcept>

using imageprocessing::filtering::ChainedFilter;
using imageprocessing::filtering::ImageFilter;
using cv::Mat;
using cv::Size;
using std::vector;
using std::shared_ptr;
using std::pair;
using std::make_shared;
using std::invalid_argument;
using std::runtime_error;

namespace imageprocessing {

shared_ptr<ImagePyramid> ImagePyramid::create(int octaveLayerCount, double minScaleFactor, double maxScaleFactor) {
	return make_shared<ImagePyramid>(octaveLayerCount, minScaleFactor, maxScaleFactor);
}

shared_ptr<ImagePyramid> ImagePyramid::createFiltered(shared_ptr<ImagePyramid> pyramid, const shared_ptr<ImageFilter>& filter) {
	auto filteredPyramid = make_shared<ImagePyramid>(pyramid);
	if (filter)
		filteredPyramid->addLayerFilter(filter);
	return filteredPyramid;
}

shared_ptr<ImagePyramid> ImagePyramid::createApproximated(
		int octaveLayerCount, double minScaleFactor, double maxScaleFactor, vector<double> lambdas) {
	return createApproximated(create(1, minScaleFactor, maxScaleFactor), octaveLayerCount, lambdas);
}

shared_ptr<ImagePyramid> ImagePyramid::createApproximated(
		shared_ptr<ImagePyramid> pyramid, int octaveLayerCount, vector<double> lambdas) {
	auto approximatePyramid = make_shared<ImagePyramid>(
			octaveLayerCount, pyramid->getMinScaleFactor(), pyramid->getMaxScaleFactor());
	approximatePyramid->setLambdas(lambdas);
	approximatePyramid->setSource(pyramid);
	return approximatePyramid;
}

ImagePyramid::ImagePyramid(int octaveLayerCount, double minScaleFactor, double maxScaleFactor) :
		octaveLayerCount(octaveLayerCount), incrementalScaleFactor(0),
		minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor),
		firstLayer(0), layers(), lambdas(), sourceImage(), sourcePyramid(), version(),
		imageFilter(make_shared<ChainedFilter>()), layerFilter(make_shared<ChainedFilter>()) {
	if (octaveLayerCount == 0)
		throw invalid_argument("ImagePyramid: the number of layers per octave must be greater than zero");
	if (minScaleFactor <= 0)
		throw invalid_argument("ImagePyramid: the minimum scale factor must be greater than zero");
	if (maxScaleFactor > 1)
		throw invalid_argument("ImagePyramid: the maximum scale factor must not exceed one");
	incrementalScaleFactor = pow(0.5, 1. / octaveLayerCount);
}

ImagePyramid::ImagePyramid(shared_ptr<ImagePyramid> pyramid, double minScaleFactor, double maxScaleFactor) :
		octaveLayerCount(pyramid->octaveLayerCount), incrementalScaleFactor(pyramid->incrementalScaleFactor),
		minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor),
		firstLayer(0), layers(), lambdas(), sourceImage(), sourcePyramid(pyramid), version(),
		imageFilter(make_shared<ChainedFilter>()), layerFilter(make_shared<ChainedFilter>()) {}

void ImagePyramid::addImageFilter(const shared_ptr<ImageFilter>& filter) {
	imageFilter->add(filter);
	if (sourcePyramid)
		sourcePyramid->addImageFilter(filter);
}

void ImagePyramid::addLayerFilter(const shared_ptr<ImageFilter>& filter) {
	layerFilter->add(filter);
}

void ImagePyramid::update(const Mat& image) {
	update(make_shared<VersionedImage>(image));
}

void ImagePyramid::update(const shared_ptr<VersionedImage>& image) {
	if (sourcePyramid) {
		sourcePyramid->update(image);
		update();
	} else {
		setSource(image);
		update();
	}
}

void ImagePyramid::setSource(const Mat& image) {
	setSource(make_shared<VersionedImage>(image));
}

void ImagePyramid::setSource(const shared_ptr<VersionedImage>& image) {
	if (octaveLayerCount == 0)
		throw invalid_argument("ImagePyramid: the number of layers per octave must be greater than zero");
	if (minScaleFactor <= 0)
		throw invalid_argument("ImagePyramid: the minimum scale factor must be greater than zero");
	sourcePyramid.reset();
	sourceImage = image;
}

void ImagePyramid::setSource(const shared_ptr<ImagePyramid>& pyramid) {
	sourceImage.reset();
	sourcePyramid = pyramid;
}

void ImagePyramid::update() {
	if (sourceImage) {
		if (version != sourceImage->getVersion()) {
			layers.clear();
			createLayers(sourceImage->getData());
			if (!layers.empty())
				firstLayer = layers.front()->getIndex();
			version = sourceImage->getVersion();
		}
	} else if (sourcePyramid) {
		if (version != sourcePyramid->version) {
			layers.clear();
			createLayers(*sourcePyramid);
			if (!layers.empty())
				firstLayer = layers.front()->getIndex();
			version = sourcePyramid->version;
		}
	}
}

void ImagePyramid::createLayers(const Mat& image) {
	Mat filteredImage = imageFilter->applyTo(image);
	for (size_t i = 0; i < octaveLayerCount; ++i) {
		double scaleFactor = pow(incrementalScaleFactor, i);
		Mat scaledImage;
		Size scaledImageSize(cvRound(filteredImage.cols * scaleFactor), cvRound(filteredImage.rows * scaleFactor));
		cv::resize(filteredImage, scaledImage, scaledImageSize, 0, 0, cv::INTER_LINEAR);
		double widthScaleFactor = static_cast<double>(scaledImage.cols) / static_cast<double>(filteredImage.cols);
		double heightScaleFactor = static_cast<double>(scaledImage.rows) / static_cast<double>(filteredImage.rows);
		if (scaleFactor <= maxScaleFactor && scaleFactor >= minScaleFactor)
			layers.push_back(make_shared<ImagePyramidLayer>(i, scaleFactor,
					widthScaleFactor, heightScaleFactor, layerFilter->applyTo(scaledImage)));
		Mat previousScaledImage = scaledImage;
		scaleFactor *= 0.5;
		for (size_t j = 1; scaleFactor >= minScaleFactor && previousScaledImage.cols > 1; ++j, scaleFactor *= 0.5) {
			pyrDown(previousScaledImage, scaledImage);
			double widthScaleFactor = static_cast<double>(scaledImage.cols) / static_cast<double>(filteredImage.cols);
			double heightScaleFactor = static_cast<double>(scaledImage.rows) / static_cast<double>(filteredImage.rows);
			if (scaleFactor <= maxScaleFactor)
				layers.push_back(make_shared<ImagePyramidLayer>(i + j * octaveLayerCount, scaleFactor,
						widthScaleFactor, heightScaleFactor, layerFilter->applyTo(scaledImage)));
			previousScaledImage = scaledImage;
		}
	}
	std::sort(layers.begin(), layers.end(), [](const shared_ptr<ImagePyramidLayer>& a, const shared_ptr<ImagePyramidLayer>& b) {
		return a->getIndex() < b->getIndex();
	});
}

void ImagePyramid::createLayers(const ImagePyramid& pyramid) {
	if (octaveLayerCount % pyramid.octaveLayerCount != 0)
		throw runtime_error(
				"ImagePyramid: octaveLayerCount must be divisible by the source pyramid's octaveLayerCount to enable approximation of layers");
	vector<shared_ptr<ImagePyramidLayer>> filteredLayers;
	filteredLayers.reserve(pyramid.layers.size());
	for (const shared_ptr<ImagePyramidLayer>& layer : pyramid.layers)
		filteredLayers.push_back(layer->createFiltered(*layerFilter));
	int layersPerOriginalLayer = octaveLayerCount / pyramid.octaveLayerCount;
	vector<double> lambdas = this->lambdas;
	if (lambdas.size() == 0)
		lambdas = estimateLambdas(filteredLayers);
	else if (!filteredLayers.empty() && filteredLayers.front()->getScaledImage().channels() != lambdas.size())
		throw runtime_error("ImagePyramid: the number number of lambdas does not match the number of channels");
	for (shared_ptr<ImagePyramidLayer>& exactLayer : filteredLayers) {
		if (exactLayer->getScaleFactor() < minScaleFactor)
			break;
		exactLayer->index *= layersPerOriginalLayer;
		if (exactLayer->getScaleFactor() <= maxScaleFactor)
			layers.push_back(exactLayer);
		Mat exactImage = exactLayer->getScaledImage();
		for (int i = 1; i < layersPerOriginalLayer; ++i) {
			double scaleFactor = pow(incrementalScaleFactor, i);
			double overallScale = exactLayer->scale * scaleFactor;
			if (overallScale >= minScaleFactor && overallScale <= maxScaleFactor) {
				Mat approximatedImage = resize(exactImage, scaleFactor, lambdas);
				double scaleFactorX = static_cast<double>(approximatedImage.cols) / static_cast<double>(exactImage.cols);
				double scaleFactorY = static_cast<double>(approximatedImage.rows) / static_cast<double>(exactImage.rows);
				double overallScaleX = exactLayer->scaleX * scaleFactor;
				double overallScaleY = exactLayer->scaleY * scaleFactor;
				layers.push_back(make_shared<ImagePyramidLayer>(exactLayer->getIndex() + i,
						overallScale, overallScaleX, overallScaleY, approximatedImage));
			}
		}
	}
}

vector<double> ImagePyramid::estimateLambdas(const vector<shared_ptr<ImagePyramidLayer>>& layers) const {
	if (layers.size() < 2)
		throw runtime_error("ImagePyramid: at least two pyramid layers are needed to estimate the lambdas");
	if (layers.size() == 2)
		return estimateLambdas(*layers[0], *layers[1]);
	else // layers.size() > 2
		return estimateLambdas(*layers[1], *layers[2]);
}

vector<double> ImagePyramid::estimateLambdas(const ImagePyramidLayer& layer1, const ImagePyramidLayer& layer2) const {
	vector<double> channelMeans1 = computeChannelMeans(layer1.getScaledImage());
	vector<double> channelMeans2 = computeChannelMeans(layer2.getScaledImage());
	vector<double> channelRatios = computeChannelRatios(channelMeans1, channelMeans2);
	double scaleFactorRatio = layer1.getScaleFactor() / layer2.getScaleFactor();
	return computeLambdas(channelRatios, scaleFactorRatio);
}

vector<double> ImagePyramid::computeChannelMeans(const Mat& image) const {
	vector<Mat> channels;
	cv::split(image, channels);
	vector<double> means(channels.size());
	for (int i = 0; i < channels.size(); ++i)
		means[i] = cv::mean(channels[i])[0];
	return means;
}

vector<double> ImagePyramid::computeChannelRatios(const vector<double>& channelMeans1, const vector<double>& channelMeans2) const {
	vector<double> ratios(channelMeans1.size());
	for (size_t i = 0; i < channelMeans1.size(); ++i)
		ratios[i] = channelMeans1[i] / channelMeans2[i];
	return ratios;
}

vector<double> ImagePyramid::computeLambdas(const vector<double>& channelRatios, double scaleFactorRatio) const {
	vector<double> lambdas(channelRatios.size());
	for (int ch = 0; ch < channelRatios.size(); ++ch)
		lambdas[ch] = -std::log(channelRatios[ch]) / std::log(scaleFactorRatio);
	return lambdas;
}

Mat ImagePyramid::resize(const Mat& image, double scaleFactor, const vector<double>& lambdas) const {
	Size scaledSize(cvRound(image.cols * scaleFactor), cvRound(image.rows * scaleFactor));
	vector<Mat> channels;
	cv::split(image, channels);
	vector<Mat> resizedChannels(channels.size());
	for (size_t i = 0; i < channels.size(); ++i) {
		cv::resize(channels[i], resizedChannels[i], scaledSize, 0, 0, cv::INTER_LINEAR);
		resizedChannels[i] *= std::pow(scaleFactor, -lambdas[i]);
	}
	Mat resizedImage;
	cv::merge(resizedChannels, resizedImage);
	return resizedImage;
}

const shared_ptr<ImagePyramidLayer> ImagePyramid::getLayer(int index) const {
	int realIndex = index - firstLayer;
	if (realIndex < 0 || realIndex >= (int)layers.size())
		return shared_ptr<ImagePyramidLayer>();
	return layers[realIndex];
}

const shared_ptr<ImagePyramidLayer> ImagePyramid::getLayer(double scaleFactor) const {
	double power = log(scaleFactor) / log(incrementalScaleFactor);
	return getLayer(static_cast<int>(std::round(power)));
}

} /* namespace imageprocessing */
