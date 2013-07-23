/*
 * Hdf5Utils.hpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */
#pragma once

#ifndef HDF5UTILS_HPP_
#define HDF5UTILS_HPP_

#include "render2/Mesh.hpp"

#ifdef WIN32	// This is a shitty hack...  find out what the proper way to do this is. Probably include the hdf5.tar.gz in our cmake project. Bzw... without cpp is maybe correct, and my windows-installation is wrong?
	#include "cpp/H5Cpp.h"
#else
	#include "H5Cpp.h"
#endif

#include "opencv2/core/core.hpp"

#include <vector>

// Todo: Class with static methods? Or just functions? I don't know which method is better.

namespace render {
	namespace utils {

		class Hdf5Utils
		{
		public:
			static H5::H5File openFile(const std::string filename);
			static H5::Group openPath(H5::H5File& file, const std::string& path);

			static cv::Mat readMatrixFloat(const H5::CommonFG& fg, std::string name);
			static void readMatrixInt(const H5::CommonFG& fg, std::string name, cv::Mat& matrix);
			static void readVector(const H5::CommonFG& fg, std::string name, std::vector<float>& vector);
			static std::string readString(const H5::CommonFG& fg, std::string name);

			static bool existsObjectWithName(const H5::CommonFG& fg, const std::string& name);

		};

	} /* namespace utils */
} /* namespace render */

#endif /* HDF5UTILS_HPP_ */
