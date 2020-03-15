#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cross_detector.h"
#include "support.h"
#include <chrono>

using namespace cv;
using namespace std;

int main() {
	const char* default_file = "C:\\Users\\dimat\\Downloads\\cross_detect_data_set_src\\img0002.jpg";
	Mat image, gray_img;
	image = imread(default_file, IMREAD_COLOR);   // Read the file
	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	cv::Rect myROI(Point(0, image.rows / 5), Point(image.cols, image.rows));
	image = image(myROI);

	cv::cvtColor(image, gray_img, COLOR_BGR2GRAY);   // Read the file

	using namespace std::chrono;
	milliseconds start_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

	std::vector<cv::Mat> int_images = support::get_integral_images(gray_img);
	
	cross_algo::grid_t grid;
	cross_algo::growing_point_sets_t res_grow_up = cross_algo::growing_up(int_images, image, grid);

	//correct_rails(grid, res_grow_up, image);

	milliseconds diff = duration_cast<milliseconds>(system_clock::now().time_since_epoch()) - start_time;

	cross_algo::draw_growing_point_sets(res_grow_up, image);
	std::cout << "time: " << diff.count() << "ms" << std::endl;
	
	imshow("Original", image);                   // Show our image inside it.
	imshow("Gray", gray_img);                   // Show our image inside it.

	waitKey(0);                                          // Wait for a keystroke in the window
	return 0;
}