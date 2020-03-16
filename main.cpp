#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cross_detector.h"
#include <chrono>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	std::string file_name = "C:\\Users\\dimat\\Downloads\\cross_detect_data_set_src\\img0002.jpg";
	if (argc < 2)
	{
		std::cout << "Wrong arguments count, program accept one argument: " <<
			"path to input file, for example: C:\\Users\\dimat\\Downloads\\cross_detect_data_set_src\\img0002.jpg" << std::endl;

		return -1;
	}
	else
		file_name = std::string(argv[1]);

	Mat image, gray_img;
	image = imread(file_name, IMREAD_COLOR);   // Read the file
	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find input file" << endl;
		return -1;
	}

	//crop 1/5 part of image
	cv::Rect myROI(Point(0, image.rows / 5), Point(image.cols, image.rows));
	image = image(myROI);

	cv::cvtColor(image, gray_img, COLOR_BGR2GRAY);

	using namespace std::chrono;

	milliseconds start_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	
	CrossDetector cross_detector(gray_img, image);
	cross_detector.detect_crosses(true);

	milliseconds diff = duration_cast<milliseconds>(system_clock::now().time_since_epoch()) - start_time;
	std::cout << "time: " << diff.count() << "ms" << std::endl;
	
	imshow("Original", image);                   // Show our image inside it.
	imshow("Gray", gray_img);                   // Show our image inside it.

	std::cout << "press any button to to close program" << std::endl;
	waitKey(0);                                          // Wait for a keystroke in the window
	return 0;
}