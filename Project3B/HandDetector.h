#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class HandDetector
{
public:
	HandDetector();
	~HandDetector() = default;
	void SetMaskF(cv::Mat& frame, cv::Mat& mask);
	void DetectContours(cv::Mat& frame, cv::Mat& dst);
	void DetectContours(cv::Mat& frame, cv::Mat& result_hsv, cv::Mat& result_masked);
	void DetectContoursOnSamples();
	void ShowSamples();

private:
	std::array<int, 3> hsv_lower;
	std::array<int, 3> hsv_upper;
	std::vector<int> hull_index;
	std::vector<cv::Mat> samples;
	cv::Mat frame_hsv;
	cv::Mat mask_f;
	cv::Rect rect_area;
	cv::Rect hull_rect;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::vector<std::vector<cv::Point>>> sample_contours;
	std::vector<cv::Point> hulls;
};

double findPointsDistance(cv::Point a, cv::Point b);
std::vector<cv::Point> compactOnNeighborhoodMedian(std::vector<cv::Point> points, double max_neighbor_distance);
double findPointsDistanceOnX(cv::Point a, cv::Point b);
std::vector<cv::Point> findClosestOnX(std::vector<cv::Point> points, cv::Point pivot);
double findAngle(cv::Point a, cv::Point b, cv::Point c);
bool isFinger(cv::Point a, cv::Point b, cv::Point c,
	          double limit_angle_inf, double limit_angle_sup,
	          cv::Point palm_center, double min_distance_from_palm);
void drawVectorPoints(cv::Mat image, std::vector<cv::Point> points, cv::Scalar color, bool with_numbers);