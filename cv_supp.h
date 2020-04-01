#pragma once
#ifndef CV_SUPP
#define CV_SUPP
#include <vector>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

namespace cv_supp {

	struct gradient_img {
		cv::Mat mag;
		cv::Mat angle;
	};

	using hog_vec_t = std::vector<double>;

	gradient_img get_gradients(const cv::Mat& img);
	std::vector<cv::Mat> get_integral_images(const cv::Mat& img);
	hog_vec_t get_hog(const cv::Point2i& left_high_hog_point, int hog_size, const std::vector<cv::Mat>& integral_images);
	double chi_squared(const hog_vec_t& hog1, const hog_vec_t& hog2);
	double intersect_hogs(const hog_vec_t& hog1, const hog_vec_t& hog2);
	bool hog_has_vertical_edge(const hog_vec_t& hog_vec);
    float get_line_cos(cv::Point pt1, cv::Point pt2);
}

#endif