#include "HandDetector.h"

HandDetector::HandDetector() : hsv_lower({0, 30, 60}), hsv_upper({20, 150, 240}),
                               rect_area(cv::Rect(170, 90, 300, 300)) {}

void HandDetector::DetectContours(cv::Mat& frame, cv::Mat& result_hsv, cv::Mat& result_masked)
{
	// create mask
	mask_f = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	cv::rectangle(mask_f, rect_area, cv::Scalar(255), -1);

	cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
	cv::inRange(frame_hsv, hsv_lower, hsv_upper, result_hsv);
	cv::bitwise_and(result_hsv, mask_f, result_masked);

	cv::medianBlur(result_hsv, result_hsv, 3);
	cv::medianBlur(result_masked, result_masked, 3);

	cv::findContours(result_masked, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	// draw detect area and contours
	cv::rectangle(frame, rect_area, cv::Scalar(0, 0, 255));
	for (int i = 0; i < contours.size(); i++)
	{
		double area_c = cv::contourArea(contours[i]);

		if (area_c < 1e4 || 8e4 < area_c) { continue; }

		cv::drawContours(frame, contours, i, cv::Scalar(0, 255, 0), 1, 8);
	}
}
