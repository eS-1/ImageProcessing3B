#include "HandDetector.h"


HandDetector::HandDetector()
{
	bgSubtractor = cv::createBackgroundSubtractorKNN();
	samples.push_back(cv::imread("hands/paper.JPG"));
	samples.push_back(cv::imread("hands/rock.JPG"));
	samples.push_back(cv::imread("hands/scissors.JPG"));
	samples_gray.push_back(cv::imread("hands/paper.JPG", 0));
	samples_gray.push_back(cv::imread("hands/rock.JPG", 0));
	samples_gray.push_back(cv::imread("hands/scissors.JPG", 0));
}


void HandDetector::BGSubtract(cv::Mat frame)
{
	bgSubtractor->apply(frame, fore_ground_mask);
	MatchShapes(fore_ground_mask);
	cv::imshow("bg_mask", fore_ground_mask);
}


void HandDetector::ShowSamples()
{
	int num = 0;
	for (auto i : samples)
	{
		std::string sam = "sample" + std::to_string(num);
		cv::imshow(sam, i);
		num++;
	}
}


void HandDetector::MatchShapes(cv::Mat& frame)
{
	int i = 0;
	int y = 40;
	int start_x = frame.cols / 2 - 150;
	int start_y = frame.rows / 2 - 150;
	cv::Scalar color_white = cv::Scalar(255, 255, 255);
	cv::Rect roi = cv::Rect(cv::Point(start_x, start_y), cv::Size(300, 300));
	cv::Mat frame_trim = frame(roi);
	cv::Point point = cv::Point(30, y);
	for (auto samp : samples_gray)
	{
		double matchLevel = cv::matchShapes(frame_trim, samp, cv::CONTOURS_MATCH_I1, 0);
		std::string text = std::to_string(i) + ": " + std::to_string(matchLevel);
		cv::putText(frame, text, point, cv::FONT_HERSHEY_PLAIN, 3, color_white);
		i++;
		y += 30;
		point = cv::Point(30, y);
	}
	cv::rectangle(frame, roi, color_white);
}
