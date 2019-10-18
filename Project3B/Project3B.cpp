#include <opencv2/opencv.hpp>

std::vector<int> HSV_LOWER = {0, 30, 60};
std::vector<int> HSV_UPPER = { 20, 150, 255 };

const std::string MASK = "Mask";
const std::string FRAME = "Frame";
const std::string HSV = "HSV";
const std::string RESULT = "Result";


void FrameRangeHSV(cv::Mat& frame, cv::Mat& result)
{
	cv::Mat frame_hsv, frame_ranged;
	cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
	cv::inRange(frame_hsv, HSV_LOWER, HSV_UPPER, frame_ranged);
	cv::medianBlur(frame_ranged, result, 3);
}


int main()
{
	cv::VideoCapture cap(0);

	if (!cap.isOpened())
	{
		printf("Can't find camera.");
		return -1;
	}

	cv::Mat frame, frame_hsv;
	std::vector<std::vector<cv::Point>> contours;

	while (cap.read(frame))
	{
		FrameRangeHSV(frame, frame_hsv);

		cv::findContours(frame_hsv, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

		for (int i = 0; i < contours.size(); i++)
		{
			double area_c = cv::contourArea(contours[i]);

			if (area_c < 1e3 || 1e5 < area_c) { continue; }

			cv::drawContours(frame, contours, i, cv::Scalar(0, 255, 0), 1, 8);
		}

		cv::imshow(FRAME, frame);
		cv::imshow(HSV, frame_hsv);

		const int key = cv::waitKey(1);
		if (key == 'q')
		{
			break;
		}
		else if (key == 'c')
		{
			cv::imwrite("frame.png", frame);
		}
	}
	cv::destroyAllWindows();
	return 0;
}
