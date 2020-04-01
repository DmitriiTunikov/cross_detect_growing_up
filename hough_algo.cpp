//
// Created by dmitrii on 01.04.2020.
//

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "hough_algo.h"
#include "cv_supp.h"

using namespace cv;

std::vector<cv::Mat> hough_algo::get_cropped_images(const cv::Mat &image, int crop_count) {
    std::vector<cv::Mat> res;

    for (int i = 1; i <= crop_count; i++) {
        int y_max = image.rows * i / crop_count;
        int y_min = y_max - image.rows / crop_count;

        cv::Rect crop(Point(0, y_min), Point(image.cols, y_max));

        line(image, Point(0, y_min), Point(image.cols, y_min), Scalar(255, 0, 0), 2);

        res.push_back(image(crop));
    }

    return res;
}



void hough_algo::find_lines_on_cropped(std::vector<cv::Mat> &crop_images, int canny_treashhold1, int canny_treashhold2) {
    for (int i = 0; i < crop_images.size(); i++) {
        cv::Mat& cur_image = crop_images[i];

        //find edges by canny
//        Mat gx, gx_abs, vertical_edges;
//        Sobel(cur_image, gx, CV_64F, 1, 0, 1);
//        convertScaleAbs(gx, gx_abs );
//        addWeighted( gx_abs, 0, gx_abs, 1, 0, vertical_edges );

        cv::Mat canny;
        cv::Canny(cur_image, canny, canny_treashhold1, canny_treashhold2);

        //find lines by hough
        std::vector<Vec2f> lines; // will hold the results of the detection
        HoughLines(canny, lines, 1, CV_PI/180, 20, 0, 0); // runs the actual detection

        //draw lines
        draw_lines(lines, cur_image);
    }
}

void hough_algo::draw_lines(const std::vector<cv::Vec2f> &lines, cv::Mat &image) {
    for( size_t j = 0; j < lines.size(); j++)
    {
        float rho = lines[j][0], theta = lines[j][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));

        if (abs(cv_supp::get_line_cos(pt1, pt2)) > 0.9)
            continue;

        line( image, pt1, pt2, Scalar(0,0,255), 1, LINE_8);
    }
}

void hough_algo::find_lines(cv::Mat &image, int crop_count, int canny_treashhold1, int canny_treashhold2) {
    std::vector<cv::Mat> croped_images = get_cropped_images(image, crop_count);

    find_lines_on_cropped(croped_images, canny_treashhold1, canny_treashhold2);
}
