/*
 * ExactFhogExtractor.hpp
 *
 *  Created on: Jan 12, 2017
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_EXTRACTION_EXACTFHOGEXTRACTOR_HPP_
#define IMAGEPROCESSING_EXTRACTION_EXACTFHOGEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/filtering/FhogFilter.hpp"

namespace imageprocessing {
namespace extraction {

/**
 * Feature extractor that exactly extracts FHOG features from the given patch. This is in contrast to feature
 * extractors that work on image pyramids, where usually scale as well as position are only approximately
 * represented by the extracted patch.
 */
class ExactFhogExtractor : public FeatureExtractor {
public:

	/**
	 * Constructs a new exact FHOG extractor.
	 *
	 * @param[in] fhogFilter Filter that computes the FHOG descriptors of the resized image region.
	 * @param[in] windowWidth Width of the extracted window in cells.
	 * @param[in] windowHeight Height of the extracted window in cells.
	 */
	ExactFhogExtractor(std::shared_ptr<filtering::FhogFilter> fhogFilter, int windowWidth, int windowHeight);

	using FeatureExtractor::update;

	void update(std::shared_ptr<VersionedImage> image) override;

	std::shared_ptr<Patch> extract(int centerX, int centerY, int width, int height) const override;

private:

	/**
	 * Extracts a rectangular window from the image.
	 *
	 * @param[in] bounds Bounds of the window.
	 * @return Rectangular sub-image.
	 */
	cv::Mat getWindow(cv::Rect bounds) const;

	/**
	 * Resizes the image window to the fixed size necessary for feature extraction.
	 *
	 * @param[in] window Image window to resize.
	 * @return Image window of fixed size.
	 */
	cv::Mat toFixedSize(cv::Mat window) const;

	/**
	 * Creates the look-up table for the image indices that are used to retrieve the patch data.
	 *
	 * @param[in] imageSize The size of the image (width or height).
	 * @param[in] patchStart The first patch index inside the image (x or y).
	 * @param[in] patchSize The size of the patch (width or height).
	 * @return A vector mapping the patch indices to image indices.
	 */
	std::vector<int> createIndexLut(int imageSize, int patchStart, int patchSize) const;

	/**
	 * Creates the patch data by copying values from the image.
	 *
	 * @param[in] image The image to take the values from.
	 * @param[in] rowIndices The mappings from patch row indices to image row indices.
	 * @param[in] colIndices The mappings from patch column indices to image column indices.
	 * @return The patch data.
	 */
	template<class T>
	cv::Mat createPatchData(const cv::Mat& image, std::vector<int>& rowIndices, std::vector<int>& colIndices) const {
		cv::Mat patch(rowIndices.size(), colIndices.size(), image.type());
		for (size_t patchY = 0; patchY < rowIndices.size(); ++patchY) {
			T* patchRow = patch.ptr<T>(patchY);
			const T* imageRow = image.ptr<T>(rowIndices[patchY]);
			for (size_t patchX = 0; patchX < colIndices.size(); ++patchX)
				patchRow[patchX] = imageRow[colIndices[patchX]];
		}
		return patch;
	}

	std::shared_ptr<filtering::FhogFilter> fhogFilter; ///< Filter that computes the FHOG descriptors of the resized image region.
	double widthFactor;  ///< Scale factor for increasing the patch width before extraction to capture surrounding cells.
	double heightFactor; ///< Scale factor for increasing the patch height before extraction to capture surrounding cells.
	cv::Size fixedSize; ///< Fixed window size necessary for feature extraction.
	cv::Mat image; ///< Image to extract features from.
};

} /* namespace extraction */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_EXTRACTION_EXACTFHOGEXTRACTOR_HPP_ */
