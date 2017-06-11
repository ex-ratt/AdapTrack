/*
 * DlibImageSource.hpp
 *
 *  Created on: 31.07.2015
 *      Author: poschmann
 */

#ifndef DLIBIMAGESOURCE_HPP_
#define DLIBIMAGESOURCE_HPP_

#include "imageio/AnnotatedImageSource.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/filesystem.hpp"

namespace imageio {

/**
 * Annotated image source that reads the file names and annotations from an XML file created with
 * the imglab-tool of dlib.
 */
class DlibImageSource : public AnnotatedImageSource {
public:

	/**
	 * Constructs a new dlib image source.
	 *
	 * @param[in] filename Name of the XML file.
	 */
	DlibImageSource(const std::string& filename);

	void reset();

	bool next();

	const cv::Mat getImage() const;

	std::string getName() const;

	Annotations getAnnotations() const;

private:

	boost::filesystem::path directory; ///< Image directory.
	boost::property_tree::ptree info; ///< Image and annotation information.
	boost::property_tree::ptree::const_assoc_iterator imagesBegin; ///< Iterator pointing to the first image entry.
	boost::property_tree::ptree::const_assoc_iterator imagesEnd; ///< Iterator pointing behind the last image entry.
	boost::property_tree::ptree::const_assoc_iterator imagesNext; ///< Iterator pointing to the next image entry.
	boost::filesystem::path filename; ///< Current image filename.
	cv::Mat image; ///< Current image.
	Annotations annotations; ///< Current annotations.
};

} /* namespace imageio */

#endif /* DLIBIMAGESOURCE_HPP_ */
