#include "cv_supp.h"

#define INTEGRAL_IMG_COUNT 8

using namespace cv_supp;
using namespace cv;

gradient_img cv_supp::get_gradients(const cv::Mat& img) {
	Mat gx, gy;

	Sobel(img, gx, CV_64F, 1, 0, 1);
	Sobel(img, gy, CV_64F, 0, 1, 1);

	gradient_img res;
	cartToPolar(gx, gy, res.mag, res.angle, true);

	return res;
}


std::vector<cv::Mat> cv_supp::get_integral_images(const cv::Mat& img) {
	gradient_img grad_img = get_gradients(img);

	std::vector<cv::Mat> res_as_hist;
	std::vector<cv::Mat> res;
	for (size_t i = 0; i < INTEGRAL_IMG_COUNT; i++)
	{
		res_as_hist.emplace_back(Size(img.cols, img.rows), CV_64F, Scalar(0));
		res.emplace_back(Size(img.cols, img.rows), CV_64F, Scalar(0));
	}

	double const angle_per_bin = 45;
	std::vector<std::pair<double, double>> bins{ std::pair<double, double>(360 - angle_per_bin / 2, angle_per_bin / 2) };
	for (int i = 1; i < INTEGRAL_IMG_COUNT; i++)
		bins.emplace_back(bins[i - 1].second, bins[i - 1].second + angle_per_bin);

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
	    cv::integral(res_as_hist[i], res[i]);
	}

	return res;
}



hog_vec_t cv_supp::get_hog(const cv::Point2i& left_high_hog_point, int hog_size, const std::vector<cv::Mat>& integral_images) {
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

double cv_supp::chi_squared(const hog_vec_t& hog1, const hog_vec_t& hog2) {
	assert(hog2.size() == hog1.size() && "wrong input sizes");

	double sum = 0;

	for (size_t i = 0; i < hog2.size(); i++) {
		sum += (hog1[i] - hog2[i]) * (hog1[i] - hog2[i]) / (hog1[i] + hog2[i]);
	}

	return sum;
}

double cv_supp::intersect_hogs(const hog_vec_t& hog1, const hog_vec_t& hog2) {
	assert(hog2.size() == hog1.size() && "wrong input sizes");

	double sum = 0;

	for (size_t i = 0; i < hog2.size(); i++) {
		sum += hog1[i] < hog2[i] ? hog1[i] : hog2[i];
	}

	return sum;
}

bool cv_supp::hog_has_vertical_edge(const hog_vec_t& hog_vec) {
	const double vertical_edge_treashhold = 0.8;
	const double horizontal_edge_treashhold = 0.5;

	//ignore horizontal edges
	if (hog_vec[2] > horizontal_edge_treashhold || hog_vec[6] > horizontal_edge_treashhold)
		return false;

	//ignore not vertical edges
	const double vertical_mul = 0.9;
    return hog_vec[0] > vertical_edge_treashhold || hog_vec[4] > vertical_edge_treashhold ||
           hog_vec[1] * vertical_mul > vertical_edge_treashhold ||
           hog_vec[3] * vertical_mul > vertical_edge_treashhold ||
           hog_vec[5] * vertical_mul > vertical_edge_treashhold || hog_vec[7] * vertical_mul > vertical_edge_treashhold;
}
