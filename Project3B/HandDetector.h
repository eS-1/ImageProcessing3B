#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class HandDetector
{
public:
	HandDetector();
	~HandDetector() = default;
	void DetectContours(cv::Mat& frame, cv::Mat& result_hsv, cv::Mat& result_masked);

private:
	std::vector<int> hsv_lower;
	std::vector<int> hsv_upper;
	cv::Mat frame_hsv;
	cv::Mat mask_f;
	cv::Rect rect_area;
	std::vector<std::vector<cv::Point>> contours;
};
