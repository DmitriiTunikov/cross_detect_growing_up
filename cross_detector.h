#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>
#include "support.h"

namespace cross_algo {

	class Cell {
	public:
		Cell() {}

		Cell(const cv::Point2i& p_, const cv::Point2i& grid_coord_, const int& size_, Cell* left_ = nullptr, Cell* center_ = nullptr, Cell* right_ = nullptr)
			: p(p_), size(size_), left(left_), right(right_), center(center_), accum_value(0), grid_coord(grid_coord_) {}

		cv::Point2i p;
		cv::Point2i grid_coord;
		int size;
		int accum_value;
		support::hog_vec_t hog_vec;
		std::vector<int> growing_points_sets_idxs;

		//neighbours
		Cell* left;
		Cell* right;
		Cell* center;
	};


	using grid_t = std::vector<std::vector<Cell>>;

	void draw_grid(const grid_t& grid, const cv::Mat& img) {
		for (const auto& grid_str : grid) {
			int x = 0;
			for (const auto& grid_elem : grid_str) {
				cv::rectangle(img, grid_elem.p, cv::Point(grid_elem.p.x + grid_elem.size, grid_elem.p.y + grid_elem.size), Scalar(200));
				char s[10];
				sprintf(s, "%d", x);
				cv::putText(img, s, cv::Point(grid_elem.p.x + grid_elem.size / 2, grid_elem.p.y + grid_elem.size / 2), FONT_HERSHEY_PLAIN, 0.5, Scalar(200));
				x++;
			}
		}
	}

	grid_t generate_grid(int min_size, int max_size, int w, int h) {
		grid_t res(1);

		int cur_y = 0;
		for (int x = 0, prev_count = 0; x + min_size < w; x += min_size, prev_count++)
			res[0].push_back(Cell(cv::Point2i(x, cur_y), cv::Point2i(res[0].size(), 0), min_size));

		cur_y += min_size;
		int prev_count = res[0].size();
		int cur_size = min_size;
		while (true) {
			std::vector<Cell> cur_cells;
			int prev_size = cur_size;
			cur_size = min_size + (float(max_size) - min_size) / (h - max_size) * cur_y;

			if (cur_size + cur_y > h)
				break;

			for (int x = 0; x + cur_size < w; x += cur_size)
			{
				Cell new_cell(cv::Point2i(x, cur_y), cv::Point2i(cur_cells.size(), res.size()), cur_size);

				//find neighs
				int center_neigh_x = round(float(x) / prev_size);
				if (center_neigh_x < prev_count)
					new_cell.center = &res[res.size() - 1][center_neigh_x];

				//has left neigh
				if (center_neigh_x > 0)
					new_cell.left = &res[res.size() - 1][center_neigh_x - 1];
				//has right neigh
				if (center_neigh_x  + 1 < prev_count)
					new_cell.right = &res[res.size() - 1][center_neigh_x + 1];

				cur_cells.emplace_back(new_cell);
			}

			prev_count = cur_cells.size();
			cur_y += cur_size;
			res.emplace_back(cur_cells);
		}

		return res;
	}

	using growing_point_sets_t = std::vector<std::vector<Cell>>;

	void draw_growing_point_sets(const growing_point_sets_t grow_point_sets, const cv::Mat& img) {
		for (const auto& set : grow_point_sets) {
			for (const auto& cell : set) {
				cv::rectangle(img, cell.p, cv::Point(cell.p.x + cell.size, cell.p.y + cell.size), Scalar(0,0, 0), -1);
			}
		}
	}


	int dist_between_rails(int y) {
		return 40 + 0.3 * y;
	}

	void correct_rails(grid_t& grid, growing_point_sets_t& growing_point_sets, const cv::Mat& img) {
		const int min_grow_points_count = 5;

		//find lost rails by duplicating founded growing_points as parallel
		for (const auto& growing_points : growing_point_sets) {
			if (growing_points.size() < min_grow_points_count)
				continue;

			//check is it left or rigth rail on track
			int r_count = 0;
			int l_count = 0;

			for (int k = 1; k < min_grow_points_count - 1; k++) {
				Cell cur_cell = growing_points[k];

				int xr = dist_between_rails(cur_cell.p.y) + cur_cell.p.x;
				int xl = cur_cell.p.x - dist_between_rails(cur_cell.p.y);

				int xr_idx = -1;
				int xl_idx = -1;

				if (xr < img.cols)
					xr_idx = xr / cur_cell.size;
				if (xl > 0)
					xl_idx = xl / cur_cell.size;

				int y_idx = cur_cell.grid_coord.y;
				if ((xr_idx != -1 && grid[y_idx][xr_idx].accum_value > 0) || (xr < img.cols - 1 && grid[y_idx][xr_idx + 1].accum_value > 0) || 
					(xr > 0 && grid[y_idx][xr_idx - 1].accum_value > 0)) {
					r_count++;
				}

				if ((xl_idx != -1 && grid[y_idx][xl_idx].accum_value > 0) || (xl < img.cols - 1 && grid[y_idx][xl_idx + 1].accum_value > 0) ||
					(xl > 0 && grid[y_idx][xl_idx - 1].accum_value > 0)) {
					l_count++;
				}
			}

			bool is_left_rail = r_count > l_count;

			for (int k = 1; k < growing_points.size(); k++) {
				Cell cur_cell = growing_points[k];

				int x = is_left_rail ? dist_between_rails(cur_cell.p.y) + cur_cell.p.x : cur_cell.p.x - dist_between_rails(cur_cell.p.y);

				if (x < img.cols && x > 0)
				{
					int x_idx = x / cur_cell.size;
					int y_idx = cur_cell.grid_coord.y;
					grid[y_idx][x_idx].accum_value++;
					cv::rectangle(img, grid[y_idx][x_idx].p,
						cv::Point(grid[y_idx][x_idx].p.x + grid[y_idx][x_idx].size, grid[y_idx][x_idx].p.y + grid[y_idx][x_idx].size),
						Scalar(0, 0, 255), -1);
				}
			}
		}
	}


	
	growing_point_sets_t growing_up(const std::vector<cv::Mat>& integral_images, const cv::Mat& img, grid_t& grid) {
		const int min_grid_size = 2;
		const int max_grid_size = 22;
		const double treashhold = 1.5;
		const double edge_treashhold = 0.65;

		//why integral image size more than image size?????
		grid = generate_grid(min_grid_size, max_grid_size, integral_images[0].cols, integral_images[0].rows);
		draw_grid(grid, img);

		std::vector<Cell> seeds = grid[grid.size() - 1];
		growing_point_sets_t growing_point_sets(seeds.size());
		for (int i = 0; i < seeds.size(); i++) {
			std::vector<Cell>& cur_growing_points = growing_point_sets[i];
			seeds[i].accum_value++;
			cur_growing_points.push_back(seeds[i]);
			seeds[i].growing_points_sets_idxs.push_back(i);

			int start = 0, end = 0;
			while (start <= end) {
				if (cur_growing_points[start].hog_vec.empty())
					cur_growing_points[start].hog_vec = support::get_hog(cur_growing_points[start].p, cur_growing_points[start].size, integral_images);

				std::vector<Cell*> neighs{ cur_growing_points[start].left, cur_growing_points[start].center, cur_growing_points[start].right };
				for (Cell* neigh : neighs) {
					//if neigh exist and it doesn't contains in current growing_points
					if (neigh != nullptr && 
						std::find(neigh->growing_points_sets_idxs.begin(), neigh->growing_points_sets_idxs.end(), i) == neigh->growing_points_sets_idxs.end())
					{
						if (neigh->hog_vec.empty())
							neigh->hog_vec = support::get_hog(neigh->p, neigh->size, integral_images);

						//ignore horizontal edges
						if (neigh->hog_vec[2] > edge_treashhold || neigh->hog_vec[6] > edge_treashhold)
							continue;
						
						//ignore not vertical edges
						if (neigh->hog_vec[0] < edge_treashhold && neigh->hog_vec[6] < edge_treashhold)
							continue;

						double intersection = support::intersect_hogs(cur_growing_points[start].hog_vec, neigh->hog_vec);
						//double chi_squared = support::chi_squared(cur_growing_points[start].hog_vec, neigh->hog_vec);
						if (intersection > treashhold)
						{
							neigh->accum_value++;
							cur_growing_points.push_back(*neigh);
							neigh->growing_points_sets_idxs.push_back(i);
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
			if (growing_point_sets[i + 1].size() == growing_point_sets[i].size() || growing_point_sets[i].size() == 1) {
				growing_point_sets.erase(growing_point_sets.begin() + i);
				cur_size--;
				i--;
			}
		}

		if (growing_point_sets[cur_size - 1].size() == 1)
			growing_point_sets.erase(growing_point_sets.begin() + cur_size - 1);

		return growing_point_sets;
	}
}

