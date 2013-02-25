/*
 * mergePlaygroundApp.cpp
 *
 *  Created on: 17.02.2013
 *      Author: Patrik Huber
 */

// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

#ifdef WIN32
	#include <crtdbg.h>
#endif

#ifdef _DEBUG
   #ifndef DBG_NEW
      #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
      #define new DBG_NEW
   #endif
#endif  // _DEBUG

#include "classification/RbfKernel.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/WvmClassifier.hpp"

#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "detection/SlidingWindowDetector.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/Patch.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "mat.h"
#include <iostream>
#include <sstream>

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif

using namespace std;
using namespace imageprocessing;
using namespace detection;
using namespace classification;


int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(22978);
	#endif
	
	cout << "Starting tests..." << endl;
	/*
	Mat fvp = cv::imread("D:/FeatureDetection/patchp.png");
	Mat fvn = cv::imread("D:/FeatureDetection/patchn.png");

	cv::cvtColor(fvp, fvp, CV_BGR2GRAY);
	cv::cvtColor(fvn, fvn, CV_BGR2GRAY);

	SvmClassifier* svm =  new SvmClassifier();
	svm->load("D:/FeatureDetection/config/fdetection/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--With-outnew02-HQ64SVM.mat", "D:/FeatureDetection/config/fdetection/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--ts107742-hq64_thres_0.005--with-outnew02HQ64SVM.mat");
	BinaryClassifier* myclass = svm;

	WvmClassifier* wvm =  new WvmClassifier();
	wvm->load("D:/FeatureDetection/config/fdetection/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--With-outnew02-HQ64SVM.mat", "D:/FeatureDetection/config/fdetection/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--ts107742-hq64_thres_0.005--with-outnew02HQ64SVM.mat");
	BinaryClassifier* myclassw = wvm;

	pair<bool, double> res;
	res = myclass->classify(fvp);
	cout << "f: " << res.first << ", s: " << res.second << endl;
	res = myclass->classify(fvn);
	cout << "f: " << res.first << ", s: " << res.second << endl;

	res = myclassw->classify(fvp);
	cout << "f: " << res.first << ", s: " << res.second << endl;
	res = myclassw->classify(fvn);
	cout << "f: " << res.first << ", s: " << res.second << endl;

	cout << "The end." << endl;
	delete svm;
	delete wvm;
	*/
	Mat img = cv::imread("D:/FeatureDetection/data/firstrun/ws_8.png");
	cv::namedWindow("src", CV_WINDOW_AUTOSIZE); cv::imshow("src", img);
	
	shared_ptr<ImagePyramid> pyr = make_shared<ImagePyramid>(0.02, 0.12, 0.7);
	shared_ptr<ImageFilter> imgFil = make_shared<GrayscaleFilter>();
	pyr->addImageFilter(imgFil);
	pyr->update(img);
	vector<shared_ptr<ImagePyramidLayer>> l = pyr->getLayers();
	for(unsigned int i=0; i<l.size(); ++i) {
		stringstream ss; ss << i;
		cv::namedWindow(ss.str(), CV_WINDOW_AUTOSIZE); cv::imshow(ss.str(), l[i]->getScaledImage());
	}
	shared_ptr<WvmClassifier> wvm = make_shared<WvmClassifier>();
	wvm->load("D:/FeatureDetection/config/fdetection/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--With-outnew02-HQ64SVM.mat", "D:/FeatureDetection/config/fdetection/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--ts107742-hq64_thres_0.005--with-outnew02HQ64SVM.mat");
	shared_ptr<SlidingWindowDetector> det = make_shared<SlidingWindowDetector>(wvm);
	vector<pair<shared_ptr<Patch>, pair<bool, double>>> resultingPatches = det->detect(pyr);

	Mat rgbimg = img.clone();
	std::vector<pair<shared_ptr<Patch>, pair<bool, double>>>::iterator pit = resultingPatches.begin();
	for(; pit != resultingPatches.end(); pit++) {
		cv::rectangle(rgbimg, cv::Point(pit->first->getX()-pit->first->getWidth()/2, pit->first->getY()-pit->first->getHeight()/2), cv::Point(pit->first->getX()+pit->first->getWidth()/2, pit->first->getY()+pit->first->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((pit->second.second+2.0)/2.0)   ));
	}
	cv::namedWindow("final", CV_WINDOW_AUTOSIZE); cv::imshow("final", rgbimg);
	cv::waitKey(0);

	return 0;
}