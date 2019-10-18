#include <opencv2/opencv.hpp>
#include "HandDetector.h"

std::vector<int> HSV_LOWER = { 0, 30, 60 };
std::vector<int> HSV_UPPER = { 20, 150, 255 };

const std::string MASK = "Mask";
const std::string FRAME = "Frame";
const std::string HSV = "HSV";
const std::string RESULT = "Result";


int main()
{
	cv::VideoCapture cap(0);

	if (!cap.isOpened())
	{
		printf("Can't find camera.");
		return -1;
	}

	HandDetector handDetector;
	cv::Mat frame, frame_hsv;
	std::vector<std::vector<cv::Point>> contours;

	while (cap.read(frame))
	{
		handDetector.DetectContours(frame, frame_hsv, contours);

		cv::imshow(FRAME, frame);
		cv::imshow(HSV, frame_hsv);

		const int key = cv::waitKey(1);
		if (key == 'q') { break; }
		else if (key == 'c')
		{
			cv::imwrite("images/frame.png", frame);
			cv::imwrite("images/mask.png", frame_hsv);
		}
	}
	cv::destroyAllWindows();
	return 0;
}
