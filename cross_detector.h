#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>
#include "support.h"

namespace cross_algo {

	class Cell {
	public:
		Cell() {}

		Cell(const cv::Point2i& p_, const cv::Point2i& grid_coord_, const int& size_)
			: p(p_), size(size_), accum_value(0), grid_coord(grid_coord_) {}

		cv::Point2i p;
		cv::Point2i grid_coord;

		int size;
		int accum_value;
		support::hog_vec_t hog_vec;
		std::vector<int> growing_points_sets_idxs;

		//neighbours
		std::vector<std::shared_ptr<Cell>> neighs;
		std::vector<std::shared_ptr<Cell>> nearest_neighs;
	};


	using grid_t = std::vector<std::vector<shared_ptr<Cell>>>;

	void draw_grid(const grid_t& grid, const cv::Mat& img) {
		for (const auto& grid_str : grid) {
			int x = 0;
			for (const auto& grid_elem : grid_str) {
				cv::rectangle(img, grid_elem->p, cv::Point(grid_elem->p.x + grid_elem->size, grid_elem->p.y + grid_elem->size), Scalar(200));
				//char s[10];
				//sprintf(s, "%d", x);
				//cv::putText(img, s, cv::Point(grid_elem->p.x + grid_elem->size / 2, grid_elem->p.y + grid_elem->size / 2), FONT_HERSHEY_PLAIN, 0.5, Scalar(200));
				//x++;
			}
		}
	}

	grid_t generate_grid(int min_size, int max_size, int w, int h) {
		grid_t res(1);

		int cur_y = 0;
		for (int x = 0, prev_count = 0; x + min_size < w; x += min_size, prev_count++)
			res[0].push_back(std::make_shared<Cell>(cv::Point2i(x, cur_y), cv::Point2i(res[0].size(), 0), min_size));

		cur_y += min_size;
		int cur_size = min_size;
		while (true) {
			std::vector<shared_ptr<Cell>> cur_cells;
			int prev_size = cur_size;
			cur_size = min_size + (float(max_size) - min_size) / (h - max_size) * cur_y;

			if (cur_size + cur_y > h)
				break;

			for (int x = 0; x + cur_size < w; x += cur_size)
			{
				std::shared_ptr<Cell> new_cell = std::make_shared<Cell>(cv::Point2i(x, cur_y), cv::Point2i(cur_cells.size(), res.size()), cur_size);

				//find neighs
				int center_neigh_x = round(float(x) / prev_size);

				//has left neigh
				if (center_neigh_x > 0)
					new_cell->nearest_neighs.push_back(res[res.size() - 1][center_neigh_x - 1]);

				//has center neigh
				if (center_neigh_x < res[res.size() - 1].size())
					new_cell->nearest_neighs.push_back(res[res.size() - 1][center_neigh_x]);

				//has right neigh
				if (center_neigh_x + 1 < res[res.size() - 1].size())
					new_cell->nearest_neighs.push_back(res[res.size() - 1][center_neigh_x + 1]);

				for (std::shared_ptr<Cell> new_neigh : new_cell->nearest_neighs) {
					new_cell->neighs.push_back(new_neigh);
				}

				cur_cells.emplace_back(new_cell);
			}

			cur_y += cur_size;
			res.push_back(cur_cells);
		}

		return res;
	}

	using growing_point_sets_t = std::vector<std::vector<std::shared_ptr<Cell>>>;

	void draw_growing_point_sets(const growing_point_sets_t grow_point_sets, const cv::Mat& img) {
		for (const auto& set : grow_point_sets) {
			Scalar color = Scalar(0);//rand() % 255, rand() % 255, rand() % 255);
			for (const auto& cell : set) {
				cv::rectangle(img, cell->p, cv::Point(cell->p.x + cell->size, cell->p.y + cell->size), color, -1);
			}
		}
	}

	void draw_crosses(const std::vector<Point>& crosses, const cv::Mat& img) {
		for (const auto& cur_cross : crosses) {
			cv::circle(img, cur_cross, 5, Scalar(0, 255, 0), 5);
		}
	}

	std::vector<Point> get_cross_result(grid_t& grid) {
		std::vector<Point> res;

		std::vector<std::pair<int, int>> prev_row_rails_x;
		std::vector<std::pair<int, int>> cur_row_rails_x;

		for (int y = grid.size() - 2; y >= 0; y--) {
			std::vector<std::shared_ptr<Cell>>& grid_row = grid[y];
			cur_row_rails_x.clear();
			for (int x = 0; x < grid_row.size(); x++) {
				if (grid_row[x]->accum_value != 0)
				{
					int start_x = x;
					x++;
					while (x < grid_row.size() && grid_row[x]->accum_value != 0)
						x++;

					int center_x = (x + start_x) / 2;
					cur_row_rails_x.push_back(std::pair<int, int>(center_x, 0));
				}
			}

			//need to check intersection?
			if (prev_row_rails_x.size() > 0)
			{
				//has intersection
				if (prev_row_rails_x.size() > cur_row_rails_x.size())
				{
					int intersection_x = -1;
					const int epsilon = 4;
					//find intersection point
					for (const auto& prev_row_elem : prev_row_rails_x)
					{
						for (auto& cur_row_elem : cur_row_rails_x)
						{
							if (abs(cur_row_elem.first - prev_row_elem.first) < epsilon)
							{
								cur_row_elem.second++;
								if (cur_row_elem.second > 1) {
									intersection_x = cur_row_elem.first;
									break;
								}
							}
						}
					}
					res.push_back(Point(intersection_x * grid_row[0]->size, grid_row[0]->p.y - grid_row[0]->size));
				}
				if (prev_row_rails_x.size() < cur_row_rails_x.size())
				{
					int intersection_x = -1;
					const int epsilon = 4;
					//find intersection point
					for (auto& cur_row_elem : cur_row_rails_x)
					{
						for (auto& prev_row_elem : prev_row_rails_x)
						{
							if (abs(cur_row_elem.first - prev_row_elem.first) < epsilon)
							{
								prev_row_elem.second++;
								if (prev_row_elem.second > 1) {
									intersection_x = prev_row_elem.first;
									break;
								}
							}
						}
					}
					res.push_back(Point(intersection_x * grid_row[0]->size, grid_row[0]->p.y - grid_row[0]->size));
				}
			}

			//copy current row rail to prev rails vec
			prev_row_rails_x.clear();
			prev_row_rails_x.resize(cur_row_rails_x.size());
			for (auto& cur_row_elem : cur_row_rails_x)
			{
				cur_row_elem.second = 0;
			}
			std::copy(cur_row_rails_x.begin(), cur_row_rails_x.end(), prev_row_rails_x.begin());
		}

		return res;
	}

	bool hog_has_vertical_edge(const support::hog_vec_t& hog_vec) {
		const double vertical_edge_treashhold = 0.8;
		const double horizontal_edge_treashhold = 0.5;

		//ignore horizontal edges
		if (hog_vec[2] > horizontal_edge_treashhold || hog_vec[6] > horizontal_edge_treashhold)
			return false;

		//ignore not vertical edges
		const double vertical_mul = 0.9;
		if (hog_vec[0] > vertical_edge_treashhold || hog_vec[4] > vertical_edge_treashhold ||
			hog_vec[1] * vertical_mul > vertical_edge_treashhold || hog_vec[3] * vertical_mul > vertical_edge_treashhold ||
			hog_vec[5] * vertical_mul > vertical_edge_treashhold || hog_vec[7] * vertical_mul > vertical_edge_treashhold)
			return true;
		else
			return false;

	}
	
	growing_point_sets_t growing_up(const std::vector<cv::Mat>& integral_images, const cv::Mat& img, grid_t& grid) {
		const int min_grid_size = 2;
		const int max_grid_size = 22;
		const double hog_chi_square_treashhold = 1.5;

		//why integral image size more than image size?????
		grid = generate_grid(min_grid_size, max_grid_size, integral_images[0].cols, integral_images[0].rows);
		draw_grid(grid, img);

		std::vector<std::shared_ptr<Cell>>& seeds = grid[grid.size() - 1];
		growing_point_sets_t growing_point_sets(seeds.size());
		for (int i = 0; i < seeds.size(); i++) {
			std::vector<std::shared_ptr<Cell>>& cur_growing_points = growing_point_sets[i];
			seeds[i]->accum_value++;
			cur_growing_points.push_back(seeds[i]);
			seeds[i]->growing_points_sets_idxs.push_back(i);

			int start = 0, end = 0;
			while (start <= end) {
				if (cur_growing_points[start]->hog_vec.empty())
					cur_growing_points[start]->hog_vec = support::get_hog(cur_growing_points[start]->p, cur_growing_points[start]->size, integral_images);

				if (!hog_has_vertical_edge(cur_growing_points[start]->hog_vec))
				{
					start++;
					continue;
				}
					
				for (int q = 0; q < cur_growing_points[start]->neighs.size(); q++)
				{
					std::shared_ptr<Cell> neigh = cur_growing_points[start]->neighs[q];
					if (neigh != nullptr && neigh->accum_value == 0)
					{
						if (neigh->hog_vec.empty())
							neigh->hog_vec = support::get_hog(neigh->p, neigh->size, integral_images);

						if (!hog_has_vertical_edge(neigh->hog_vec))
							continue;

						//double intersection = support::intersect_hogs(cur_growing_points[start].hog_vec, neigh->hog_vec);
						double chi_squared = support::chi_squared(cur_growing_points[start]->hog_vec, neigh->hog_vec);
						if (chi_squared < hog_chi_square_treashhold)
						{
							neigh->accum_value++;
							cur_growing_points.push_back(neigh);
							neigh->growing_points_sets_idxs.push_back(i);
							//cv::line(img, Point(cur_growing_points[start].p.x + cur_growing_points[start].size / 2,
							//	cur_growing_points[start].p.y + cur_growing_points[start].size / 2),
							//	Point(neigh->p.x + neigh->size / 2,
							//		neigh->p.y + neigh->size / 2), Scalar(0, 0, 255));
							end++;
						}
					}
				}
				start++;
			}
		}

		//remove same rails
		int cur_size = growing_point_sets.size();
		for (int i = 0; i < cur_size - 1; i++) {
			if (growing_point_sets[i + 1].size() == growing_point_sets[i].size() || growing_point_sets[i].size() < 5) {
				growing_point_sets.erase(growing_point_sets.begin() + i);
				cur_size--;
				i--;
			}
		}

		if (growing_point_sets[cur_size - 1].size() < 5)
			growing_point_sets.erase(growing_point_sets.begin() + cur_size - 1);

		return growing_point_sets;
	}
}

