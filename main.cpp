#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils/files_utils.h"
#include "cross_detector.h"
#include "hough_algo.h"
#include <chrono>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	if (argc < 2)
	{
		std::cout << "Wrong arguments count, program accept one argument: " <<
			R"(path to input file, for example: C:\Users\dimat\Downloads\cross_detect_data_set_src\img0002.jpg)" << std::endl;

		return -1;
	}
    std::string file_name = std::string(argv[1]);

	Mat image, gray_img, canny_res;
	image = imread(file_name, IMREAD_COLOR);
	if (!image.data)
	{
		cout << "Could not open or find input file" << endl;
		return -1;
	}

    cv::cvtColor(image, gray_img, COLOR_BGR2GRAY);

    //Blur the image with 5x5 Gaussian kernel
//    Mat image_blurred_with_5x5_kernel;
//    GaussianBlur(image, image_blurred_with_5x5_kernel, Size(5, 5), 0);

	int canny_treashhold1 = 150, canny_treashhold2 = 250;
    std::chrono::milliseconds start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    cv::Canny(image, canny_res, canny_treashhold1, canny_treashhold2);
    hough_algo::find_lines(image, 20, canny_treashhold1, canny_treashhold2);

    std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_time;
    std::cout << "time: " << diff.count() << "ms" << std::endl;

    imshow("original", image);
    imshow("canny", canny_res);

    waitKey(0);
/*
	//crop 1/5 part of image
	cv::Rect myROI(Point(0, image.rows / 5), Point(image.cols, image.rows));
	image = image(myROI);

    start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
	
	CrossDetector cross_detector(gray_img, image);
	cross_detector.detect_crosses(true);

	diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_time;
	std::cout << "time: " << diff.count() << "ms" << std::endl;
	
	imshow("Original", image);
	imshow("Gray", gray_img);

	std::cout << "press any button to close program" << std::endl;
	waitKey(0);*/
	return 0;
}