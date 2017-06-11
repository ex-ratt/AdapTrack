/*
 * CameraImageSource.cpp
 *
 *  Created on: 27.09.2013
 *      Author: poschmann
 */

#include "imageio/CameraImageSource.hpp"

using cv::Mat;
using std::invalid_argument;
using std::runtime_error;
using std::string;
using std::to_string;

namespace imageio {

CameraImageSource::CameraImageSource(int device) :
		device(device), capture(device), frame() {
	if (!capture.isOpened())
		throw invalid_argument("CameraImageSource: Could not open stream from device " + to_string(device));
}

CameraImageSource::~CameraImageSource() {
	capture.release();
}

void CameraImageSource::reset() {
	capture.release();
	if (!capture.open(device))
		throw runtime_error("CameraImageSource: Could not open stream from device " + to_string(device));
}

bool CameraImageSource::next() {
	return capture.read(frame);
}

const Mat CameraImageSource::getImage() const {
	return frame;
}

string CameraImageSource::getName() const {
	return "";
}

} /* namespace imageio */
