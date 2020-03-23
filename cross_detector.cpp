#include "cross_detector.h"

std::vector<Point> CrossDetector::detect_crosses(bool need_to_draw_grid)
{
	//algorithm
	m_integral_images = cv_supp::get_integral_images(m_img);

	auto res_grow_up = growing_up();

	if (need_to_draw_grid)
	{
		draw_grid();
		draw_growing_point_sets(res_grow_up);
	}
	
	std::vector<Point> cross_res = get_cross_result();
	draw_crosses(cross_res);

	return cross_res;
}

void CrossDetector::generate_grid(int max_size) {
	int low_h = (double(4)/5) * m_img.rows;

	double a = double(max_size) / (low_h  - max_size);

	int cur_size = max_size;
	int cur_y = m_img.rows;
	for (int i = 0; cur_y - cur_size >= 0 && cur_size > 1; i++)
	{
		cur_size = abs(low_h * a / ((1 + a * i) * (1 + a * i + a)));
		m_grid.emplace_back();
		for (int x = 0; x + cur_size < m_img.cols; x += cur_size)
		{
			m_grid[i].push_back(std::make_shared<Cell>(cv::Point2i(x, cur_y), cv::Point2i(m_grid[i].size(), i), cur_size));
		}
		cur_y -= cur_size;
	}

}

void CrossDetector::get_intersection_point(std::vector<Point>& res, std::vector<std::pair<int, int>>& big_vec,
	std::vector<std::pair<int, int>>& small_vec, std::shared_ptr<Cell> cell) {
	int intersection_x = -1;
	const int epsilon = 4;
	//find intersection point
	for (auto& cur_big_vec_elem : big_vec)
	{
		for (auto& prev_small_vec_elem : small_vec)
		{
			if (abs(cur_big_vec_elem.first - prev_small_vec_elem.first) < epsilon)
			{
				prev_small_vec_elem.second++;
				if (prev_small_vec_elem.second > 1) {
					intersection_x = prev_small_vec_elem.first;
					break;
				}
			}
		}
	}
	res.push_back(Point(intersection_x * cell->size, cell->p.y - cell->size));
}

std::vector<Point> CrossDetector::get_cross_result() {
	std::vector<Point> res;

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
				get_intersection_point(res, prev_row_rails_x, cur_row_rails_x, grid_row[0]);
			else if (prev_row_rails_x.size() < cur_row_rails_x.size())
				get_intersection_point(res, cur_row_rails_x, prev_row_rails_x, grid_row[0]);
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


CrossDetector::growing_point_sets_t CrossDetector::growing_up() {
	const double hog_chi_square_treashhold = 1.5;

	generate_grid(20);
	return growing_point_sets_t();
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