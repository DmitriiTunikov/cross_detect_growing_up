#include "cross_detector.h"

using namespace cv;

std::vector<Point> CrossDetector::detect_crosses(bool need_to_draw_grid) {
	//algorithm
	m_integral_images = cv_supp::get_integral_images(m_img);

	auto res_grow_up = growing_up();

	if (need_to_draw_grid) {
		draw_grid();
		draw_growing_point_sets(res_grow_up);
	}
	
	get_cross_result();
	draw_crosses(m_cross_res);

	return m_cross_res;
}

void CrossDetector::generate_grid(int min_size, int max_size) {
	m_grid.resize(1);

	int cur_y = 0;
	for (int x = 0, prev_count = 0; x + min_size < m_img.cols; x += min_size, prev_count++)
		m_grid[0].push_back(std::make_shared<Cell>(cv::Point2i(x, cur_y), cv::Point2i(m_grid[0].size(), 0), min_size));

	cur_y += min_size;
	int cur_size = min_size;
	while (true) {
		std::vector<std::shared_ptr<Cell>> cur_cells;
		int prev_size = cur_size;
		cur_size = min_size + (float(max_size) - min_size) / (m_img.rows - max_size) * cur_y;

		if (cur_size + cur_y > m_img.rows)
			break;

		for (int x = 0; x + cur_size < m_img.cols; x += cur_size)
		{
			std::shared_ptr<Cell> new_cell = std::make_shared<Cell>(cv::Point2i(x, cur_y), cv::Point2i(cur_cells.size(), m_grid.size()), cur_size);

			//find neighs
			int center_neigh_x = round(float(x) / prev_size);

			//has left neigh
			if (center_neigh_x > 0)
				new_cell->nearest_neighs.push_back(m_grid[m_grid.size() - 1][center_neigh_x - 1]);

			//has center neigh
			if (center_neigh_x < m_grid[m_grid.size() - 1].size())
				new_cell->nearest_neighs.push_back(m_grid[m_grid.size() - 1][center_neigh_x]);

			//has right neigh
			if (center_neigh_x + 1 < m_grid[m_grid.size() - 1].size())
				new_cell->nearest_neighs.push_back(m_grid[m_grid.size() - 1][center_neigh_x + 1]);

			for (std::shared_ptr<Cell> new_neigh : new_cell->nearest_neighs) {
				new_cell->neighs.push_back(new_neigh);
			}

			cur_cells.emplace_back(new_cell);
		}

		cur_y += cur_size;
		m_grid.push_back(cur_cells);
	}
}

void CrossDetector::get_intersection_points(std::vector<std::pair<int, int>>& big_vec,
                                            std::vector<std::pair<int, int>>& small_vec, std::shared_ptr<Cell> cell, bool from_big_to_small_count) {
	std::vector<int> intersection_xs;
	const int epsilon = 4;
	//find intersection point
	for (auto& cur_big_vec_elem : big_vec)
	{
		for (auto& cur_small_vec_elem : small_vec)
		{
			if (abs(cur_big_vec_elem.first - cur_small_vec_elem.first) < epsilon)
			{
				cur_small_vec_elem.second++;
				if (cur_small_vec_elem.second > 1) {
					intersection_xs.push_back(cur_small_vec_elem.first);
				}
			}
		}
	}

	int y_move = from_big_to_small_count ? -cell->size * 4 : cell->size * 4;
	for (int& inter_x : intersection_xs)
		m_cross_res.emplace_back(inter_x * cell->size + cell->size / 2, cell->p.y + y_move);
}

std::vector<Point> CrossDetector::get_cross_result() {
    m_cross_res.clear();

	std::vector<std::pair<int, int>> prev_row_rails_x;
	std::vector<std::pair<int, int>> cur_row_rails_x;

	for (int y = m_grid.size() - 2; y >= 0; y--) {
		std::vector<std::shared_ptr<Cell>>& grid_row = m_grid[y];
		cur_row_rails_x.clear();
		for (int x = 0; x < grid_row.size(); x++) {
			if (grid_row[x]->accum_value != 0)
			{
				int start_x = x;
				x++;
				int empty_cells_count = 0;
				while (x < grid_row.size())
				{
					if (grid_row[x]->accum_value != 0)
						x++;
					else if (++empty_cells_count > 2)
						break;
				}
					
				int center_x = (x + start_x) / 2;
				cur_row_rails_x.emplace_back(center_x, 0);
			}
		}

		if (!prev_row_rails_x.empty())
		{
			//has intersection
			if (prev_row_rails_x.size() > cur_row_rails_x.size())
                get_intersection_points(prev_row_rails_x, cur_row_rails_x, grid_row[0], true);
			else if (prev_row_rails_x.size() < cur_row_rails_x.size())
                get_intersection_points( cur_row_rails_x, prev_row_rails_x, grid_row[0], false);
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

	return m_cross_res;
}


CrossDetector::growing_point_sets_t CrossDetector::growing_up() {
	const int min_grid_size = 2;
	const int max_grid_size = 18;
	const double hog_chi_square_treashhold = 1.5;

	//why integral image size more than image size?????
	generate_grid(min_grid_size, max_grid_size);

	std::vector<std::shared_ptr<Cell>>& seeds = m_grid[m_grid.size() - 1];
	growing_point_sets_t growing_point_sets(seeds.size());
	for (int i = 0; i < seeds.size(); i++) {
		std::vector<std::shared_ptr<Cell>>& cur_growing_points = growing_point_sets[i];
		seeds[i]->accum_value++;
		cur_growing_points.push_back(seeds[i]);
		seeds[i]->growing_points_sets_idxs.push_back(i);

		int start = 0, end = 0;
		while (start <= end) {
			if (cur_growing_points[start]->hog_vec.empty())
				cur_growing_points[start]->hog_vec = cv_supp::get_hog(cur_growing_points[start]->p, cur_growing_points[start]->size, m_integral_images);

			if (!cv_supp::hog_has_vertical_edge(cur_growing_points[start]->hog_vec))
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
						neigh->hog_vec = cv_supp::get_hog(neigh->p, neigh->size, m_integral_images);

					if (!cv_supp::hog_has_vertical_edge(neigh->hog_vec))
						continue;

					double chi_squared = cv_supp::chi_squared(cur_growing_points[start]->hog_vec, neigh->hog_vec);
					if (chi_squared < hog_chi_square_treashhold)
					{
						neigh->accum_value++;
						cur_growing_points.push_back(neigh);
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