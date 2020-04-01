//
// Created by dmitrii on 01.04.2020.
//

#ifndef CROSSDETECTOR_HOUGH_ALGO_H
#define CROSSDETECTOR_HOUGH_ALGO_H

#include <vector>
#include <opencv2/core/mat.hpp>

namespace hough_algo {
    void find_lines(cv::Mat& image, int crop_count, int canny_treashhold1, int canny_treashhold2);

    std::vector<cv::Mat> get_cropped_images(const cv::Mat& image, int crop_count);

    void find_lines_on_cropped(std::vector<cv::Mat>& crop_images, int canny_treashhold1, int canny_treashhold2);

    void draw_lines(const std::vector<cv::Vec2f>& lines, cv::Mat& image);
};


#endif //CROSSDETECTOR_HOUGH_ALGO_H
