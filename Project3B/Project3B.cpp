#include <chrono>
#include "HandDetector.h"


const std::string MASKED = "Masked";
const std::string FRAME  = "Frame";
const std::string HSV    = "HSV";
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

	handDetector.ShowSamples();
	cap.read(frame);

	while (cap.read(frame))
	{
		auto start = std::chrono::system_clock::now();

		handDetector.detectHand(frame);

		auto end = std::chrono::system_clock::now();
		auto dur = end - start;
		auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::string msec_text = "time(milli_sec): " + std::to_string(msec);
		cv::putText(frame, msec_text, cv::Point(30, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));

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
