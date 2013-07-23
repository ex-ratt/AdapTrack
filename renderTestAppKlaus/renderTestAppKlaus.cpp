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

#include "render/MorphableModel.hpp"
#include "render/SRenderer.hpp"
#include "render/Vertex.hpp"
#include "render/Triangle.hpp"
#include "render/Camera.hpp"
#include "render/MatrixUtils.hpp"
#include "render/MeshUtils.hpp"
#include "render/Texture.hpp"
#include "render/Mesh.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"

#include <iostream>
#include <fstream>

namespace po = boost::program_options;
using namespace std;


cv::Vec4f matToColVec4f(cv::Mat m) {
	cv::Vec4f ret;
	ret[0] = m.at<float>(0, 0);
	ret[1] = m.at<float>(1, 0);
	ret[2] = m.at<float>(2, 0);
	ret[3] = m.at<float>(3, 0);
	return ret;
}

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
	copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
	return os;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(3759128);
	#endif

	std::string filename; // Create vector to hold the filenames
	
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("input-file,i", po::value<string>(), "input image")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).
				  options(desc).run(), vm);
		po::notify(vm);
	
		if (vm.count("help")) {
			cout << "[renderTestApp] Usage: options_description [options]\n";
			cout << desc;
			return 0;
		}
		if (vm.count("input-file"))
		{
			cout << "[renderTestApp] Using input images: " << vm["input-file"].as< vector<string> >() << "\n";
			filename = vm["input-file"].as<string>();
		}
	}
	catch(std::exception& e) {
		cout << e.what() << "\n";
		return 1;
	}


	render::Renderer->create();
	
	render::Mesh cube = render::utils::MeshUtils::createCube();
	render::Mesh plane = render::utils::MeshUtils::createPlane();

	//render::Mesh mmHeadL4 = render::utils::MeshUtils::readFromHdf5("D:\\model2012_l6_rms.h5");
	render::MorphableModel mmHeadL4 = render::utils::MeshUtils::readFromScm("C:\\Users\\Patrik\\Cloud\\PhD\\MorphModel\\ShpVtxModelBin.scm");

	const float& aspect = 640.0f/480.0f;

	render::Renderer->camera.setFrustum(-1.0f*aspect, 1.0f*aspect, 1.0f, -1.0f, 0.1f, 100.0f);

	vector<int> vertexIds;
	vertexIds.push_back(177); // left-eye-left - right.eye.corner_outer
	vertexIds.push_back(181); // left-eye-right - right.eye.corner_inner
	vertexIds.push_back(614); // right-eye-left - left.eye.corner_inner
	vertexIds.push_back(610); // right-eye-right - left.eye.corner_outer
	vertexIds.push_back(398); // mouth-left - right.lips.corner
	vertexIds.push_back(812); // mouth-right - left.lips.corner
	vertexIds.push_back(); // bridge of the nose - 
	vertexIds.push_back(114); // nose-tip - center.nose.tip
	vertexIds.push_back(); // nasal septum - 
	vertexIds.push_back(); // left-alare - 
	vertexIds.push_back(); // right-alare - 

	for (int pitch = -30; pitch <= 30; ++pitch) {
		render::Renderer->camera.update(1);

		cv::Mat vt = render::Renderer->constructViewTransform();
		cv::Mat pt = render::Renderer->constructProjTransform();
		cv::Mat viewProjTransform = pt * vt;


		render::Renderer->setMesh(&mmHeadL4.mesh);
		cv::Mat headWorld = render::utils::MatrixUtils::createScalingMatrix(1.0f/100.0f, 1.0f/100.0f, 1.0f/100.0f);
		cv::Mat headWorldRot = render::utils::MatrixUtils::createRotationMatrixX(pitch*(CV_PI/180));
		cv::Mat mvp_3dmm = viewProjTransform * headWorld * headWorldRot;

		//cv::Mat myvec = (cv::Mat_<float>(4,1) << 
		//	0.5f, 0.5f,	-0.5f, 1.0f);

		cv::Mat myvec = cv::Mat(mmHeadL4.mesh.vertex[0].position);

		cv::Mat res = mvp_3dmm * myvec;
		// project from 4D to 2D window position with depth value in z coordinate
		cv::Vec4f position = matToColVec4f(res);
		position = position / position[3];	// divide by w
		cv::Mat tmp = render::Renderer->getWindowTransform() * cv::Mat(position);	// places the vec as a column in the matrix
		position = matToColVec4f(tmp);


		cv::namedWindow("renderOutput");
		cv::imshow("renderOutput", render::Renderer->getRendererImage());

		float speed = 1.0f;
		float mouseSpeed = 0.10f;
		cv::Vec3f eye = render::Renderer->camera.getEye();
		float deltaTime = 1.0f;

		std::cout << "verticalAngle: " << render::Renderer->camera.verticalAngle << std::endl;
		std::cout << "horizontalAngle: " << render::Renderer->camera.horizontalAngle << std::endl;


		char c = (char)cv::waitKey(0);

	}

	// loop end - measure the time here

	cv::imwrite("colorBuffer.png", render::Renderer->getRendererImage());
	cv::imwrite("depthBuffer.png", render::Renderer->getRendererDepthBuffer());

	return 0;
}

// TODO: All those vec3 * mat4 cases... Do I have to add the homogeneous coordinate and make it vec4 * mat4, instead of how I'm doing it now? Difference?