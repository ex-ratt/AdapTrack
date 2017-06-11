/*
 * SingleAnnotationSource.cpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#include "imageio/SingleAnnotationSource.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

using cv::Rect;
using std::string;
using std::invalid_argument;

namespace imageio {

SingleAnnotationSource::SingleAnnotationSource(const string& filename) : annotations(), index(-1) {
	string line;
	char separator;
	std::ifstream file(filename.c_str());
	if (!file.is_open())
		throw invalid_argument("SingleAnnotationSource: file \"" + filename + "\" cannot be opened");
	Rect bounds;
	while (file.good()) {
		if (!std::getline(file, line))
			break;
		// read values from line
		std::istringstream lineStream(line);
		bool nonSpaceSeparator = !std::all_of(line.begin(), line.end(), [](char ch) {
			std::locale loc;
			return std::isdigit(ch, loc) || std::isspace(ch, loc) || '-' == ch;
		});
		lineStream >> bounds.x;
		if (nonSpaceSeparator)
			lineStream >> separator;
		lineStream >> bounds.y;
		if (nonSpaceSeparator)
			lineStream >> separator;
		lineStream >> bounds.width;
		if (nonSpaceSeparator)
			lineStream >> separator;
		lineStream >> bounds.height;
		if (bounds.width == 0 || bounds.height == 0)
			annotations.push_back(Annotations{});
		else
			annotations.push_back(Annotations{std::vector<Annotation>{Annotation(bounds)}});
	}
}

void SingleAnnotationSource::reset() {
	index = -1;
}

bool SingleAnnotationSource::next() {
	index++;
	return index < static_cast<int>(annotations.size());
}

Annotations SingleAnnotationSource::getAnnotations() const {
	if (index >= 0 && index < static_cast<int>(annotations.size()))
		return annotations[index];
	return Annotations{};
}

} /* namespace imageio */
