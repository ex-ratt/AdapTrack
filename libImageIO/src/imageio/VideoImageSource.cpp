/*
 * VideoImageSource.cpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#include "imageio/VideoImageSource.hpp"
#include <stdexcept>

using cv::Mat;
using std::string;
using std::invalid_argument;
using std::runtime_error;

namespace imageio {

VideoImageSource::VideoImageSource(string video) :
		video(video), capture(video), frame(), frameCounter(0) {
	if (!capture.isOpened())
		throw invalid_argument("VideoImageSource: Could not open video file '" + video + "'");
}

VideoImageSource::~VideoImageSource() {
	capture.release();
}

void VideoImageSource::reset() {
	capture.release();
	if (!capture.open(video))
		throw runtime_error("VideoImageSource: Could not open video file '" + video + "'");
	frameCounter = 0;
}

bool VideoImageSource::next() {
	++frameCounter;
	return capture.read(frame);
}

const Mat VideoImageSource::getImage() const {
	return frame;
}

string VideoImageSource::getName() const {
	return "frame" + std::to_string(frameCounter);
}

} /* namespace imageio */
