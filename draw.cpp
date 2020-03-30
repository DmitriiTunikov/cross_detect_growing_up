#include "cross_detector.h"

using namespace cv;

void CrossDetector::draw_growing_point_sets(const growing_point_sets_t& grow_point_sets) {
	for (const auto& set : grow_point_sets) {
		Scalar color = Scalar(0);//rand() % 255, rand() % 255, rand() % 255);
		for (const auto& cell : set) {
			cv::rectangle(m_img_color, cell->p, cv::Point(cell->p.x + cell->size, cell->p.y + cell->size), color, -1);
		}
	}
}

void CrossDetector::draw_crosses(const std::vector<Point>& crosses) {
	for (const auto& cur_cross : crosses) {
		cv::circle(m_img, cur_cross, 5, Scalar(0, 255, 0), 5);
	}
}

void CrossDetector::draw_grid() {
	for (const auto& grid_str : m_grid) {
		int x = 0;
		for (const auto& grid_elem : grid_str) {
			cv::rectangle(m_img_color, grid_elem->p, cv::Point(grid_elem->p.x + grid_elem->size, grid_elem->p.y + grid_elem->size), Scalar(200));
			//char s[10];
			//sprintf(s, "%d", x);
			//cv::putText(img, s, cv::Point(grid_elem->p.x + grid_elem->size / 2, grid_elem->p.y + grid_elem->size / 2), FONT_HERSHEY_PLAIN, 0.5, Scalar(200));
			//x++;
		}
	}
}