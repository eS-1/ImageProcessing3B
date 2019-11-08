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
	cv::Mat frame, frame_hsv, frame_masked, mask_f;

	// create mask
	cap.read(frame);
	handDetector.SetMaskF(frame, mask_f);
	handDetector.DetectContoursOnSamples();

	while (cap.read(frame))
	{
		handDetector.DetectContours(frame, frame_hsv, frame_masked);

		cv::imshow(FRAME, frame);
		// cv::imshow(MASKED, frame_masked);
		// cv::imshow(HSV, frame_hsv);
		// handDetector.ShowSamples();

		const int key = cv::waitKey(1);
		if (key == 'q') { break; }
		else if (key == 'c')
		{
			cv::imwrite("images/frame.png", frame);
			cv::imwrite("images/hsv.png", frame_hsv);
			cv::imwrite("images/masked.png", frame_masked);
		}
	}
	cv::destroyAllWindows();
	return 0;
}
