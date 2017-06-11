/*
 * DirectoryImageSource.cpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#include "imageio/DirectoryImageSource.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdexcept>

using boost::filesystem::exists;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;
using std::copy;
using std::sort;
using std::runtime_error;

namespace imageio {

DirectoryImageSource::DirectoryImageSource(const string& directory) : files(), index(-1) {
	path dirpath(directory);
	if (!exists(dirpath))
		throw runtime_error("DirectoryImageSource: Directory '" + directory + "' does not exist.");
	if (!is_directory(dirpath))
		throw runtime_error("DirectoryImageSource: '" + directory + "' is not a directory.");
	copy(directory_iterator(dirpath), directory_iterator(), back_inserter(files));
	vector<string> imageExtensions = { "bmp", "dib", "pbm", "pgm", "ppm", "sr", "ras", "jpeg", "jpg", "jpe", "jp2", "png", "tiff", "tif" };
	auto newFilesEnd = std::remove_if(files.begin(), files.end(), [&](const path& file) {
		string extension = file.extension().string();
		if (extension.size() > 0)
			extension = extension.substr(1);
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
		return std::none_of(imageExtensions.begin(), imageExtensions.end(), [&](const string& imageExtension) {
			return imageExtension == extension;
		});
	});
	files.erase(newFilesEnd, files.end());
	sort(files.begin(), files.end());
}

void DirectoryImageSource::reset() {
	index = -1;
}

bool DirectoryImageSource::next() {
	index++;
	return index < static_cast<int>(files.size());
}

const Mat DirectoryImageSource::getImage() const {
	if (index < 0 || index >= static_cast<int>(files.size()))
		return Mat();
	Mat image = cv::imread(files[index].string(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
		throw runtime_error("DirectoryImageSource: image '" + files[index].string() + "' could not be loaded");
	return image;
}

string DirectoryImageSource::getName() const {
	if (index < 0 || index >= static_cast<int>(files.size()))
		return "";
	return files[index].filename().string();
}

} /* namespace imageio */
