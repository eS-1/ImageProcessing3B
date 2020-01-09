#include "HandDetector.h"


HandDetector::HandDetector() : hsv_lower({ 0, 60, 80 }), hsv_higher({ 28, 160, 240 })
{
	samples.push_back(cv::imread("hands/paper.JPG"));
	samples.push_back(cv::imread("hands/rock.JPG"));
	samples.push_back(cv::imread("hands/scissors.JPG"));
	for (auto i : samples)
	{
		cv::Mat gray;
		cv::cvtColor(i, gray, cv::COLOR_BGR2GRAY);
		samples_gray.push_back(gray);
	}
}

void HandDetector::detectHand(cv::Mat &frame)
{
	cv::Mat gauss, hsv, detected, frame_gray;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::vector<cv::Point>> con_subset;

	cv::GaussianBlur(frame, gauss, cv::Size(5, 5), 0);
	cv::cvtColor(gauss, hsv, cv::COLOR_BGR2HSV);
	cv::inRange(hsv, hsv_lower, hsv_higher, hsv_mask);
	cv::morphologyEx(hsv_mask, hsv_mask, cv::MORPH_OPEN, cv::Mat());

	cv::findContours(hsv_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	for (int i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours.at(i));
		if (area > 7e3 && area < 6e4)
		{
			con_subset.push_back(contours.at(i));
		}
	}
	hsv_mask = cv::Mat::zeros(hsv_mask.size(), CV_8UC1);
	if (con_subset.empty())
	{
		cv::imshow("detected hand", hsv_mask);
		return;
	}
	cv::drawContours(hsv_mask, con_subset, 0, cv::Scalar(255), -1);

	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
	cv::bitwise_and(frame_gray, hsv_mask, detected);

	MatchShapes(detected);

	cv::imshow("detected hand", detected);
}

void HandDetector::ShowSamples()
{
	int num = 0;
	for (auto samp : samples_gray)
	{
		std::string str = "sample" + std::to_string(num);
		cv::imshow(str, samp);
		num++;
	}
}

void HandDetector::MatchShapes(cv::Mat &frame)
{
	int i = 0;
	int y = 40;
	cv::Point point = cv::Point(30, y);
	for (auto samp : samples_gray)
	{
		double match_level1 = cv::matchShapes(frame, samp, cv::CONTOURS_MATCH_I1, 0);
		double match_level2 = cv::matchShapes(frame, samp, cv::CONTOURS_MATCH_I2, 0);
		double match_level3 = cv::matchShapes(frame, samp, cv::CONTOURS_MATCH_I3, 0);
		double match_average = (match_level1 + match_level2 + match_level3) / 3;
		std::string text1 = std::to_string(i) + ": " + std::to_string(match_average);
		cv::putText(frame, text1, point, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255));
		i++;
		y += 30;
		point = cv::Point(30, y);
	}
}
