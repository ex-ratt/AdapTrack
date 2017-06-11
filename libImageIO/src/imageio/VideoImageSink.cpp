/*
 * VideoImageSink.cpp
 *
 *  Created on: 18.12.2012
 *      Author: poschmann
 */

#include "imageio/VideoImageSink.hpp"
#include <stdexcept>

using cv::Size;
using cv::Mat;
using std::string;

namespace imageio {

VideoImageSink::VideoImageSink(const string filename, double fps, int fourcc) :
		filename(filename), fps(fps), fourcc(fourcc), writer() {}

VideoImageSink::~VideoImageSink() {
	writer.release();
}

void VideoImageSink::add(const Mat& image) {
	if (!writer.isOpened()) {
		if (!writer.open(filename, fourcc, fps, Size(image.cols, image.rows)))
			throw std::runtime_error("VideoImageSink: Could not write video file '" + filename + "'");
	}
	writer << image;
}

} /* namespace imageio */
