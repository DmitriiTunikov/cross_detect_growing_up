#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "opencv2/imgproc.hpp"
#include <chrono>
using namespace cv;
using namespace std;
/*
struct math_point {
	int x;
	int y;

	double size() const {
		return sqrt(x * x + y * y);
	}
};

struct math_vec {
	math_point p1;
	math_point p2;
};

double get_scalar_mul(const math_point& v1, const math_point& v2) {
	return v1.x * v2.x + v1.y * v2.y;
}

double get_cos_between_lines(const math_point& v1, const math_point& v2) {
	return get_scalar_mul(v1, v2) / v1.size() * v2.size();
}


/*
//SIFT
int main(int argc, const char* argv[])
{
	const cv::Mat input = cv::imread("C:\\Users\\dimat\\Downloads\\cross_detect_data_set_src\\img0000.jpg", IMREAD_GRAYSCALE); //Load as grayscale
	try {
		cv::xfeatures2d::SiftFeatureDetector detector;
		std::vector<cv::KeyPoint> keypoints;
		detector.detect(input, keypoints);

		// Add results to image and save.
		cv::Mat output;
		cv::drawKeypoints(input, keypoints, output);

		imshow("Input image", input);
		waitKey();
	}
	catch (const cv::Exception& e) {
		std::cout << e.what() << std::endl;
		return -1;
	}
	
	return 0;

//HOUGH

int main(int argc, char** argv)
{
	// Declare the output variables
	Mat dst, cdst, cdstP, gaus;
	const char* default_file = "C:\\Users\\dimat\\Downloads\\cross_detect_data_set_src\\img0000.jpg";
	const char* filename = argc >= 2 ? argv[1] : default_file;
	// Loads an image
	Mat src = imread(filename, IMREAD_GRAYSCALE);

	// Check if image is loaded fine
	if (src.empty()) {
		printf(" Error opening image\n");
		printf(" Program Arguments: [image_name -- default %s] \n", default_file);
		return -1;
	}
	cv::Rect myROI(0, src.size().height * 0.2, src.size().width, src.size().height * 0.8);
	src = src(myROI);
	imshow("Source", src);

	using namespace std::chrono;
	milliseconds start_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	
	//make gause filtration for smoothing image
	GaussianBlur(src, gaus, Size(9, 9), 4, 4);
	imshow("Gause", gaus);

	//find edges by canny algorithm
	Canny(gaus, dst, 30, 40);
	imshow("Canny", dst);

	int edge_point_count = 0;
	try {
		for (int y = dst.rows / 2; y < dst.rows / 2 + 1; y++)
		{
			for (int x = 0; x < dst.cols; x++)
			{
				std::cout << y << " " << x << std::endl;
				Vec3b colour = dst.at<Vec3b>(Point(x, y));
				if (colour.val[0] == 255 && colour.val[1] == 255 && colour.val[2] == 255)
					edge_point_count++;
			}
		}
	}
	catch (std::exception& e) {
		cout << e.what() << std::endl;
	}

	// Copy edges to the images that will display the results in BGR
	cvtColor(src, cdst, COLOR_GRAY2BGR);
	cdstP = cdst.clone();
	
	// Probabilistic Line Transform
	vector<Vec4i> linesP; // will hold the results of the detection
	HoughLinesP(dst, linesP, 10, CV_PI / 90, 50, 50, 50); // runs the actual detection
	milliseconds diff = duration_cast<milliseconds>(system_clock::now().time_since_epoch()) - start_time;
	std::cout << "time: " << diff.count() << "ms" << std::endl;

	// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		if (abs(get_cos_between_lines(math_point{ l[2] - l[0], l[3] - l[1] }, math_point{ 1, 0 })) > 0.8)
			continue;

		line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
	}

	//imshow("10 on 10", croped_src);
	imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
	// Wait and Exit
	waitKey();
	return 0;
}*/

/*
harris
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

const char* source_window = "Source image";
const char* corners_window = "Corners detected";

/// Function header
void cornerHarris_demo(int, void*);

int main(int argc, char** argv)
{
	/// Load source image and convert it to gray
	const char* filename = "C:\\Users\\dimat\\Pictures\\1.png";
	src = imread(filename, 1);
	cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

	/// Create a window and a trackbar
	namedWindow(source_window, cv::WINDOW_AUTOSIZE);
	createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
	imshow(source_window, src);

	cornerHarris_demo(0, 0);

	waitKey(0);
	return(0);
}

void cornerHarris_demo(int, void*)
{

	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result
	namedWindow(corners_window, cv::WINDOW_AUTOSIZE);
	imshow(corners_window, dst_norm_scaled);
}*/


/* SIFT 
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv;
using namespace cv::features2d;
using std::cout;
using std::endl;
int main(int argc, char* argv[])
{
	CommandLineParser parser(argc, argv, "{@input | box.png | input image}");
	Mat src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_GRAYSCALE);
	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		cout << "Usage: " << argv[0] << " <Input image>" << endl;
		return -1;
	}
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints;
	detector->detect(src, keypoints);
	//-- Draw keypoints
	Mat img_keypoints;
	drawKeypoints(src, keypoints, img_keypoints);
	//-- Show detected (drawn) keypoints
	imshow("SURF Keypoints", img_keypoints);
	waitKey();
	return 0;
}*/