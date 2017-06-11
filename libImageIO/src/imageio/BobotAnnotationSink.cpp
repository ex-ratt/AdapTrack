/*
 * BobotAnnotationSink.cpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#include "imageio/BobotAnnotationSink.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Rect;
using std::invalid_argument;
using std::string;
using std::shared_ptr;
using std::runtime_error;

namespace imageio {

BobotAnnotationSink::BobotAnnotationSink(const string& videoFilename, int imageWidth, int imageHeight) :
		videoFilename(videoFilename), imageWidth(imageWidth), imageHeight(imageHeight), imageSource(), output(), index(0) {
	output.setf(std::ios_base::fixed, std::ios_base::floatfield);
	output.precision(6);
}

BobotAnnotationSink::BobotAnnotationSink(const string& videoFilename, shared_ptr<ImageSource> imageSource) :
		videoFilename(videoFilename), imageWidth(0), imageHeight(0), imageSource(imageSource), output(), index(0) {
	output.setf(std::ios_base::fixed, std::ios_base::floatfield);
	output.precision(6);
}

BobotAnnotationSink::~BobotAnnotationSink() {
	close();
}

bool BobotAnnotationSink::isOpen() {
	return output.is_open();
}

void BobotAnnotationSink::open(const string& filename) {
	if (isOpen())
		throw runtime_error("BobotAnnotationSink: sink is already open");
	output.open(filename);
	output << videoFilename << '\n';
}

void BobotAnnotationSink::close() {
	output.close();
	index = 0;
}

void BobotAnnotationSink::add(Annotations annotations) {
	if (annotations.annotations.size() > 1)
		throw invalid_argument("BobotAnnotationSink: there must be at most one annotation per frame");
	if (!isOpen())
		throw runtime_error("BobotAnnotationSink: sink is not open");
	if (imageSource) {
		Mat image = imageSource->getImage();
		imageWidth = image.cols;
		imageHeight = image.rows;
	}
	if (annotations.annotations.empty()) {
		output << index++ << " 0 0 0 0\n";
	} else {
		Rect bounds = annotations.annotations[0].bounds;
		float x = bounds.x / imageWidth;
		float y = bounds.y / imageHeight;
		float width = bounds.width / imageWidth;
		float height = bounds.height / imageHeight;
		output << index++ << ' ' << x << ' ' << y << ' ' << width << ' ' << height << '\n';
	}
}

} /* namespace imageio */
