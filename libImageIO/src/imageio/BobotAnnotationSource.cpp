/*
 * BobotAnnotationSource.cpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#include "imageio/BobotAnnotationSource.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

using cv::Mat;
using cv::Point;
using cv::Rect;
using cv::Rect_;
using std::string;
using std::shared_ptr;
using std::invalid_argument;

namespace imageio {

BobotAnnotationSource::BobotAnnotationSource(const string& filename, int imageWidth, int imageHeight) :
		imageWidth(imageWidth), imageHeight(imageHeight), imageSource(), videoFilename(), positions(), index(-1) {
	readPositions(filename);
}

BobotAnnotationSource::BobotAnnotationSource(const string& filename, shared_ptr<ImageSource> imageSource) :
		imageWidth(0), imageHeight(0), imageSource(imageSource), videoFilename(), positions(), index(-1) {
	readPositions(filename);
}

void BobotAnnotationSource::readPositions(const string& filename) {
	string name;
	string line;
	std::ifstream file(filename.c_str());
	if (!file.is_open())
		throw invalid_argument("BobotAnnotationSource: file \"" + filename + "\" cannot be opened");
	if (file.good())
		std::getline(file, videoFilename);
	Rect_<float> position;
	while (file.good()) {
		if (!std::getline(file, line))
			break;
		// read values from line
		std::istringstream lineStream(line);
		lineStream >> name;
		lineStream >> position.x;
		lineStream >> position.y;
		lineStream >> position.width;
		lineStream >> position.height;
		positions.push_back(position);
	}
}

const string& BobotAnnotationSource::getVideoFilename() const {
	return videoFilename;
}

void BobotAnnotationSource::reset() {
	index = -1;
}

bool BobotAnnotationSource::next() {
	index++;
	return index < static_cast<int>(positions.size());
}

Annotations BobotAnnotationSource::getAnnotations() const {
	Annotations annotations;
	if (index >= 0 && index < static_cast<int>(positions.size())) {
		if (imageSource) {
			const Mat& image = imageSource->getImage();
			imageWidth = image.cols;
			imageHeight = image.rows;
		}
		const Rect_<float>& relativePosition = positions[index];
		if (relativePosition.width != 0 && relativePosition.height != 0) {
			Rect bounds(
					Point(std::round(relativePosition.x * imageWidth), std::round(relativePosition.y * imageHeight)),
					Point(std::round((relativePosition.x + relativePosition.width) * imageWidth), std::round((relativePosition.y + relativePosition.height) * imageHeight)));
			annotations.annotations.emplace_back(bounds);
		}
	}
	return annotations;
}

} /* namespace imageio */
