#include "HandDetector.h"

HandDetector::HandDetector() : hsv_lower({0, 30, 60}), hsv_upper({20, 150, 255}) {}

void HandDetector::DetectContours(cv::Mat& frame, cv::Mat& result, std::vector<std::vector<cv::Point>>& conts)
{
	cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
	cv::inRange(frame_hsv, hsv_lower, hsv_upper, result);
	cv::medianBlur(result, result, 5);

	cv::findContours(result, conts, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	for (int i = 0; i < conts.size(); i++)
	{
		double area_c = cv::contourArea(conts[i]);

		if (area_c < 2e4 || 4e4 < area_c) { continue; }

		cv::drawContours(frame, conts, i, cv::Scalar(0, 255, 0), 1, 8);
	}
}