/*
 * SingleAnnotationSink.cpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#include "imageio/SingleAnnotationSink.hpp"
#include <stdexcept>

using cv::Rect;
using std::invalid_argument;
using std::string;
using std::runtime_error;

namespace imageio {

SingleAnnotationSink::SingleAnnotationSink(size_t precision) : output() {
	output.setf(std::ios_base::fixed, std::ios_base::floatfield);
	output.precision(precision);
}

SingleAnnotationSink::SingleAnnotationSink(const string& filename, size_t precision) : output(filename) {
	output.setf(std::ios_base::fixed, std::ios_base::floatfield);
	output.precision(precision);
}

SingleAnnotationSink::~SingleAnnotationSink() {
	close();
}

bool SingleAnnotationSink::isOpen() {
	return output.is_open();
}

void SingleAnnotationSink::open(const string& filename) {
	if (isOpen())
		throw runtime_error("SingleLandmarkSink: sink is already open");
	output.open(filename);
}

void SingleAnnotationSink::close() {
	output.close();
}

void SingleAnnotationSink::add(Annotations annotations) {
	if (annotations.annotations.size() > 1)
		throw invalid_argument("SingleAnnotationSink: there must be at most one annotation per frame");
	if (!isOpen())
		throw runtime_error("SingleLandmarkSink: sink is not open");
	if (annotations.annotations.empty()) {
		output << "0 0 0 0\n";
	} else {
		Rect bounds = annotations.annotations[0].bounds;
		output << bounds.x << ' ' << bounds.y << ' ' << bounds.width << ' ' << bounds.height << '\n';
	}
}

} /* namespace imageio */
