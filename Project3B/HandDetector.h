#pragma once
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

class HandDetector
{
public:
	HandDetector();
	~HandDetector() = default;
	void detectHand(cv::Mat &frame);
	void ShowSamples();
	void MatchShapes(cv::Mat &frame);

private:
	std::vector<cv::Mat> samples;
	std::vector<cv::Mat> samples_gray;
	std::array<int, 3> hsv_lower;
	std::array<int, 3> hsv_higher;
	cv::Mat hsv_mask;
};
