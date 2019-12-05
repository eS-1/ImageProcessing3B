#include <opencv2/opencv.hpp>
#include "HandDetector.h"


const std::string MASKED = "Masked";
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
	cv::Mat frame;

	cap.read(frame);
	handDetector.BGSubtract(frame);
	handDetector.ShowSamples();

	while (cap.read(frame))
	{
		handDetector.BGSubtract(frame);

		cv::imshow(FRAME, frame);

		const int key = cv::waitKey(1);
		if (key == 'q') { break; }
		else if (key == 'c')
		{
			cv::imwrite("images/frame.png", frame);
		}
	}
	cv::destroyAllWindows();
	return 0;
}
