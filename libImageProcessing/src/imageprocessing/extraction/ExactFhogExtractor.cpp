/*
 * ExactFhogExtractor.cpp
 *
 *  Created on: Jan 12, 2017
 *      Author: poschmann
 */

#include "imageprocessing/extraction/ExactFhogExtractor.hpp"
#include <stdexcept>
#include <vector>

using cv::Mat;
using cv::Point;
using cv::Rect;
using cv::Size;
using std::shared_ptr;
using std::vector;

namespace imageprocessing {
namespace extraction {

ExactFhogExtractor::ExactFhogExtractor(shared_ptr<filtering::FhogFilter> fhogFilter, int windowWidth, int windowHeight) :
		fhogFilter(fhogFilter),
		widthFactor(static_cast<double>(windowWidth + 2) / windowWidth),
		heightFactor(static_cast<double>(windowHeight + 2) / windowHeight),
		fixedSize((windowWidth + 2) * fhogFilter->getCellSize(), (windowHeight + 2) * fhogFilter->getCellSize()),
		image() {}

void ExactFhogExtractor::update(shared_ptr<VersionedImage> image) {
	this->image = image->getData();
}

shared_ptr<Patch> ExactFhogExtractor::extract(int centerX, int centerY, int width, int height) const {
	Rect bounds = Patch::computeBounds(Point(centerX, centerY), Size(width, height));
	if (bounds.x < 0 || bounds.x + bounds.width > image.cols
			|| bounds.y < 0 || bounds.y + bounds.height > image.rows)
		return shared_ptr<Patch>();
	int paddedWidth = static_cast<int>(std::round(widthFactor * width));
	int paddedHeight = static_cast<int>(std::round(heightFactor * height));
	Rect paddedBounds = Patch::computeBounds(Point(centerX, centerY), Size(paddedWidth, paddedHeight));
	if (paddedBounds.width < fixedSize.width || paddedBounds.height < fixedSize.height)
		return shared_ptr<Patch>();
	Mat fhogWindow = fhogFilter->applyTo(toFixedSize(getWindow(paddedBounds)));
	Mat fhogData = Mat(fhogWindow, Rect(1, 1, fhogWindow.cols - 2, fhogWindow.rows - 2)).clone();
	return std::make_shared<Patch>(centerX, centerY, width, height, fhogData);
}

Mat ExactFhogExtractor::getWindow(Rect bounds) const {
	if (bounds.x >= 0 && bounds.y >= 0 && bounds.x + bounds.width <= image.cols && bounds.y + bounds.height <= image.rows) {
		return Mat(image, bounds);
	} else { // window is partially outside the image
		vector<int> rowIndices = createIndexLut(image.rows, bounds.y, bounds.height);
		vector<int> colIndices = createIndexLut(image.cols, bounds.x, bounds.width);
		if (image.type() == CV_8UC1)
			return createPatchData<uchar>(image, rowIndices, colIndices);
		else if (image.type() == CV_8UC3)
			return createPatchData<cv::Vec3b>(image, rowIndices, colIndices);
		else if (image.type() == CV_32FC1)
			return createPatchData<float>(image, rowIndices, colIndices);
		else if (image.type() == CV_32FC3)
			return createPatchData<cv::Vec3f>(image, rowIndices, colIndices);
		else
			throw std::runtime_error("ExactFhogExtractor: the type of the image has to be CV_8UC1, CV_8UC3, CV_32FC1, or CV_32FC3");
	}
}

Mat ExactFhogExtractor::toFixedSize(Mat window) const {
	Mat resizedWindow;
	cv::resize(window, resizedWindow, fixedSize, 0, 0, cv::INTER_AREA);
	return resizedWindow;
}

vector<int> ExactFhogExtractor::createIndexLut(int imageSize, int patchStart, int patchSize) const {
	vector<int> indices(patchSize);
	for (int patchIndex = 0; patchIndex < patchSize; ++patchIndex) {
		int imageIndex = patchStart + patchIndex;
		if (imageIndex < 0)
			imageIndex = -imageIndex - 1;
		else if (imageIndex >= imageSize)
			imageIndex = 2 * imageSize - imageIndex - 1;
		indices[patchIndex] = imageIndex;
	}
	return indices;
}

} /* namespace extraction */
} /* namespace imageprocessing */
