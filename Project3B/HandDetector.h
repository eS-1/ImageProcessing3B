#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class HandDetector
{
public:
	HandDetector();
	~HandDetector() = default;
	void BGSubtract(cv::Mat frame);
	void ShowSamples();
	void MatchShapes(cv::Mat& frame);

private:
	std::vector<cv::Mat> samples;
	std::vector<cv::Mat> samples_gray;
	cv::Ptr<cv::BackgroundSubtractor> bgSubtractor;
	cv::Mat fore_ground_mask;
};
