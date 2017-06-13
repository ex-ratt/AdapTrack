/*
 * SingleTracker.cpp
 *
 *  Created on: 11.05.2017
 *      Author: poschmann
 */

#include "stacktrace.hpp"
#include "imageio/DlibImageSource.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tracking/SingleTracker.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace cv;
using namespace imageio;
using namespace std;
using namespace std::chrono;

int main(int argc, char **argv) {
	if (argc != 7) {
		cout << "usage: " << argv[0] << " annotations bins cellsize targetsize padding adaptation" << endl;
		cout << "  annotation: XML-file that contains image paths and annotations in dlib format" << endl;
		cout << "  bins: number of bins in unsigned orientation histogram" << endl;
		cout << "  cellsize: size of the square FHOG cells in pixels" << endl;
		cout << "  targetsize: size of the target in FHOG cells (larger one of width or height)" << endl;
		cout << "  padding: number of cells around the previous target position that is searched for the new position" << endl;
		cout << "  adaptation: weight of the new SVM parameters (between zero and one)" << endl;
		return EXIT_FAILURE;
	}
	string annotationFile = argv[1];
	int binCount = stoi(argv[2]);
	int cellSize = stoi(argv[3]);
	int targetSize = stoi(argv[4]);
	int padding = stoi(argv[5]);
	double adaptationRate = stof(argv[6]);
	tracking::SingleTracker tracker(binCount, cellSize, targetSize, padding, 1.05, 10, adaptationRate);
	Scalar color(0, 255, 0);
	int thickness = 2;

	bool run = true;
	bool pause = false;
	bool initialized = false;
	int frameCount = 0;
	duration<double> iterationTimeSum(0);
	DlibImageSource images(annotationFile);
	while (run && images.next()) {
		++frameCount;
		Mat frame = images.getImage();
		vector<Rect> annotations = images.getAnnotations().positiveAnnotations();
		if (annotations.size() > 1)
			throw invalid_argument("only one annotation per frame is permitted");
		Rect target;
		milliseconds iterationTime;
		if (!initialized) {
			if (annotations.size() == 1) {
				steady_clock::time_point iterationStart = steady_clock::now();
				target = tracker.init(frame, annotations[0], false);
				steady_clock::time_point iterationEnd = steady_clock::now();
				milliseconds iterationTime = duration_cast<milliseconds>(iterationEnd - iterationStart);
				initialized = target.area() > 0;
				if (initialized) {
					iterationTimeSum += iterationTime;
					if (frameCount > 1)
						cout << "skipped " << (frameCount - 1) << " frames before initialization" << endl;
					frameCount = 1;
				}
			}
		} else {
			steady_clock::time_point iterationStart = steady_clock::now();
			target = tracker.update(frame);
			steady_clock::time_point iterationEnd = steady_clock::now();
			milliseconds iterationTime = duration_cast<milliseconds>(iterationEnd - iterationStart);
			iterationTimeSum += iterationTime;
		}
		if (initialized) {
			double iterationFps = static_cast<double>(frameCount) / iterationTimeSum.count();
			cout << fixed << setprecision(1);
			cout << frameCount << ": ";
			cout << iterationTime.count() << " ms -> ";
			cout << iterationFps << " fps" << endl;
			rectangle(frame, target, color, thickness);
			imshow("Frame", frame);
			char c = (char)waitKey(pause ? 0 : 25);
			if (c == 'q') {
				run = false;
			} else if (c == 'p') {
				pause = !pause;
			} else if (c == 'r') {
				initialized = false;
				frameCount = 0;
				iterationTimeSum = milliseconds::zero();
			}
		}
	}

	return EXIT_SUCCESS;
}
