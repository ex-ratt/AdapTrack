/*
 * DlibImageSource.cpp
 *
 *  Created on: 31.07.2015
 *      Author: poschmann
 */

#include "imageio/DlibImageSource.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/optional.hpp"
#include "boost/property_tree/xml_parser.hpp"
#include <stdexcept>

using std::string;

namespace imageio {

DlibImageSource::DlibImageSource(const string& filename) {
	directory = boost::filesystem::path(filename).parent_path();
	boost::property_tree::xml_parser::read_xml(filename, info);
	const boost::property_tree::ptree& images = info.get_child("dataset.images");
	auto range = images.equal_range("image");
	imagesBegin = range.first;
	imagesEnd = range.second;
	imagesNext = imagesBegin;
}

void DlibImageSource::reset() {
	imagesNext = imagesBegin;
}

bool DlibImageSource::next() {
	if (imagesNext == imagesEnd)
		return false;
	string imageFilename = imagesNext->second.get<string>("<xmlattr>.file");
	boost::filesystem::path imageFilepath = directory;
	imageFilepath /= imageFilename;
	filename = imageFilepath;
	image = cv::imread(imageFilepath.string(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
		throw std::runtime_error("image '" + imageFilepath.string() + "' could not be loaded");
	annotations.annotations.clear();
	int objectCount = 0;
	int ignoreCount = 0;
	auto boxesRange = imagesNext->second.equal_range("box");
	for (auto it = boxesRange.first; it != boxesRange.second; ++it) {
		int top = it->second.get<int>("<xmlattr>.top");
		int left = it->second.get<int>("<xmlattr>.left");
		int width = it->second.get<int>("<xmlattr>.width");
		int height = it->second.get<int>("<xmlattr>.height");
		boost::optional<bool> ignore = it->second.get_optional<bool>("<xmlattr>.ignore");
		annotations.annotations.emplace_back(cv::Rect(left, top, width, height), ignore && *ignore);
	}
	++imagesNext;
	return true;
}

const cv::Mat DlibImageSource::getImage() const {
	return image;
}

string DlibImageSource::getName() const {
	return filename.filename().string();
}

Annotations DlibImageSource::getAnnotations() const {
	return annotations;
}

} /* namespace imageio */
