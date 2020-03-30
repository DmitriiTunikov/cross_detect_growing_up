#pragma once
#ifndef CROSS_DETECTOR
#define CROSS_DETECTOR
#include <opencv2/imgproc/imgproc.hpp>
#include <utility>
#include <vector>
#include <algorithm>
#include "cv_supp.h"

class CrossDetector {
public:
	class Cell {
	public:
		Cell() = default;

		Cell(cv::Point2i  p_, cv::Point2i  grid_coord_, const int& size_)
			: p(std::move(p_)), size(size_), accum_value(0), grid_coord(std::move(grid_coord_)) {}

		cv::Point2i p;
		cv::Point2i grid_coord;

        int size{};
		int accum_value{};
		cv_supp::hog_vec_t hog_vec;
		std::vector<int> growing_points_sets_idxs;

		//neighbours
		std::vector<std::shared_ptr<Cell>> neighs;
		std::vector<std::shared_ptr<Cell>> nearest_neighs;
	};

	using grid_t = std::vector<std::vector<std::shared_ptr<Cell>>>;
	using growing_point_sets_t = std::vector<std::vector<std::shared_ptr<Cell>>>;


	CrossDetector(cv::Mat img_gray, cv::Mat image_color) : m_img(std::move(img_gray)), m_img_color(std::move(image_color)) {}
	std::vector<cv::Point> detect_crosses(bool need_to_draw_grid = false);
private:
	cv::Mat m_img;
    std::vector<cv::Point> m_cross_res;
	cv::Mat m_img_color;
	grid_t m_grid;
	std::vector<cv::Mat> m_integral_images;

	//drawing functions
	void draw_grid();
	void draw_growing_point_sets(const growing_point_sets_t& grow_point_sets);
	void draw_crosses(const std::vector<cv::Point>& crosses);

	//algo functions
	void generate_grid(int min_size, int max_size);
	std::vector<cv::Point> get_cross_result();

	void get_intersection_points(std::vector<std::pair<int, int>>& big_vec,
                                 std::vector<std::pair<int, int>>& small_vec, std::shared_ptr<Cell> cell, bool from_big_to_small_count);

	growing_point_sets_t growing_up();
};

#endif