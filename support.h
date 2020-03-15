#pragma once
#include <vector>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#define INTEGRAL_IMG_COUNT 8
using namespace cv;
using namespace std;

namespace support {

	struct gradient_img {
		Mat mag;
		Mat angle;
	};

	gradient_img get_gradients(const cv::Mat& img) {
		Mat gx, gy;

		Sobel(img, gx, CV_64F, 1, 0, 1);
		Sobel(img, gy, CV_64F, 0, 1, 1);

		gradient_img res;
		cartToPolar(gx, gy, res.mag, res.angle, 1);

		return res;
	}

	std::vector<cv::Mat> get_integral_images(const cv::Mat& img) {
		gradient_img grad_img = get_gradients(img);

		std::vector<cv::Mat> res_as_hist;
		std::vector<cv::Mat> res;
		for (size_t i = 0; i < INTEGRAL_IMG_COUNT; i++)
		{
			res_as_hist.emplace_back(Size(img.cols, img.rows), CV_64F, Scalar(0));
			res.emplace_back(Size(img.cols, img.rows), CV_64F, Scalar(0));
		}

		double const angle_per_bin = 45;
		std::vector<std::pair<double, double>> bins{std::pair<double, double>(360 - angle_per_bin / 2, angle_per_bin / 2) };
		for (int i = 1; i < INTEGRAL_IMG_COUNT; i++)
			bins.push_back(std::pair<double, double>(bins[i - 1].second, bins[i - 1].second + angle_per_bin));

		//count magnitude for all bins
		for (int y = 0; y < img.rows; y++)
			for (int x = 0; x < img.cols; x++) {
				double cur_angle = grad_img.angle.at<double>(y, x);
				int bin_num = 0;
				if (cur_angle > bins[0].first || cur_angle < bins[0].second)
					bin_num = 0;
				else {
					for (bin_num = 1; bin_num < bins.size(); bin_num++)
					{
						std::pair<double, double> bin = bins[bin_num];
						if (cur_angle > bin.first && cur_angle < bin.second) {
							break;
						}
					}
				}

				res_as_hist[bin_num].at<double>(y, x) = grad_img.mag.at<double>(y, x);
			}

		//count integral image for all bins
		for (size_t i = 0; i < INTEGRAL_IMG_COUNT; i++)
		{
			res[i].at<double>(0, 0) = res_as_hist[i].at<double>(0, 0);
			
			for (int y = 1; y < img.rows; y++)
			{
				res[i].at<double>(y, 0) = res[i].at<double>(y - 1, 0) + res_as_hist[i].at<double>(y, 0);
			}
			for (int x = 1; x < img.cols; x++)
			{
				res[i].at<double>(0, x) = res[i].at<double>(0, x - 1) + res_as_hist[i].at<double>(0, x);
			}
				
	
			for (int y = 1; y < img.rows; y++)
				for (int x = 1; x < img.cols; x++) {
					res[i].at<double>(y, x) = res[i].at<double>(y - 1, x) + res[i].at<double>(y, x - 1) - res[i].at<double>(y - 1, x - 1) + res_as_hist[i].at<double>(y, x);
				}
		}
			//cv::integral(res_as_hist[i], res[i]);

		return res;
	}

	using hog_vec_t = std::vector<double>;
	hog_vec_t get_hog(const cv::Point2i& left_high_hog_point, int hog_size, const std::vector<cv::Mat>& integral_images) {
		hog_vec_t res(INTEGRAL_IMG_COUNT);

		double max = 0;
		//count hist
		for (size_t i = 0; i < INTEGRAL_IMG_COUNT; i++) {
			res[i] = integral_images[i].at<double>(left_high_hog_point.y, left_high_hog_point.x) +
				integral_images[i].at<double>(left_high_hog_point.y + hog_size, left_high_hog_point.x + hog_size) -
				integral_images[i].at<double>(left_high_hog_point.y, left_high_hog_point.x + hog_size) -
				integral_images[i].at<double>(left_high_hog_point.y + hog_size, left_high_hog_point.x);

			if (res[i] > max)
				max = res[i];
		}

		//normalize hist
		if (max != 0)
		{
			for (size_t i = 0; i < INTEGRAL_IMG_COUNT; i++) {
				res[i] /= max;
			}
		}

		return res;
	}

	double chi_squared(const hog_vec_t& hog1, const hog_vec_t& hog2) {
		assert(hog2.size() == hog1.size() && "wrong input sizes");

		double sum = 0;

		for (size_t i = 0; i < hog2.size(); i++) {
			sum += (hog1[i] - hog2[i]) * (hog1[i] - hog2[i]) / (hog1[i] + hog2[i]);
		}

		return sum;
	}

	double intersect_hogs(const hog_vec_t& hog1, const hog_vec_t& hog2) {
		assert(hog2.size() == hog1.size() && "wrong input sizes");

		double sum = 0;

		for (size_t i = 0; i < hog2.size(); i++) {
			sum += hog1[i] < hog2[i] ? hog1[i] : hog2[i];
		}

		return sum;
	}
}