#include "HandDetector.h"


HandDetector::HandDetector() : hsv_lower({0, 30, 60}), hsv_upper({20, 150, 240}),
                               rect_area(cv::Rect(170, 90, 300, 300))
{
	samples.push_back(cv::imread("hands/paper.JPG"));
	samples.push_back(cv::imread("hands/rock.JPG"));
	samples.push_back(cv::imread("hands/scissors.JPG"));
}


// create mask
void HandDetector::SetMaskF(cv::Mat& frame, cv::Mat& mask)
{
	mask_f = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	mask = mask_f;
	cv::rectangle(mask_f, rect_area, cv::Scalar(255), -1);
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


void HandDetector::DetectContoursOnSamples()
{
	std::vector<cv::Mat> samples_mask;
	std::array<int, 3> hsv_lower_s = { 0, 0, 0 };
	std::array<int, 3> hsv_upper_s = { 100, 110, 240 };

	for (auto i : samples)
	{
		cv::Mat hsv, ranged;
		cv::cvtColor(i, hsv, cv::COLOR_BGR2HSV);
		cv::inRange(hsv, hsv_lower_s, hsv_upper_s, ranged);
		cv::bitwise_not(ranged, ranged);
		samples_mask.push_back(ranged);
	}

	int num = 0;
	for (auto i : samples_mask)
	{
		std::string samp = "sample" + std::to_string(num);
		DetectContours(i, samples[num]);
		cv::imshow(samp, samples[num]);
		num++;
	}
}


void HandDetector::DetectContours(cv::Mat& frame, cv::Mat& dst)
{
	if (frame.empty()) { return; }

	cv::findContours(frame, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	if (contours.size() <= 0) { return; }

	int biggest_index = -1;
	double biggest_area = 0.0;
	for (int i = 0; i < contours.size(); i++)
	{
		double area_c = cv::contourArea(contours[i]);

		if (area_c < 1e3 || 1e5 < area_c) { continue; }

		if (area_c > biggest_area)
		{
			biggest_area = area_c;
			biggest_index = i;
		}

		cv::drawContours(frame, contours, i, cv::Scalar(0, 255, 0), 1, 8);
	}

	if (biggest_index < 0) { return; }

	cv::convexHull(cv::Mat(contours[biggest_index]), hulls);
	cv::convexHull(cv::Mat(contours[biggest_index]), hull_index);
	hull_rect = cv::boundingRect(cv::Mat(hulls));

	std::vector<cv::Vec4i> defects;
	if (hull_index.size() > 3)
	{
		cv::convexityDefects(cv::Mat(contours[biggest_index]), hull_index, defects);
	}
	else { return; }

	std::vector<cv::Point> start_points;
	std::vector<cv::Point> far_points;

	cv::Point center_rect((hull_rect.tl().x + hull_rect.br().x) / 2,
		(hull_rect.tl().y + hull_rect.br().y) / 2);

	for (int i = 0; i < defects.size(); i++)
	{
		start_points.push_back(contours[biggest_index][defects[i].val[0]]);
		if (findPointsDistance(contours[biggest_index][defects[i].val[2]], center_rect) < hull_rect.height * 0.3)
		{
			far_points.push_back(contours[biggest_index][defects[i].val[2]]);
		}
	}

	std::vector<cv::Point> filtered_start_points = compactOnNeighborhoodMedian(start_points, hull_rect.height * 0.05);
	std::vector<cv::Point> filtered_far_points = compactOnNeighborhoodMedian(far_points, hull_rect.height * 0.05);

	std::vector<cv::Point> filtered_finger_points;
	if (filtered_far_points.size() > 1)
	{
		std::vector<cv::Point> finger_points;
		for (int i = 0; i < filtered_start_points.size(); i++) {
			std::vector<cv::Point> closest_points = findClosestOnX(filtered_far_points, filtered_start_points[i]);

			if (isFinger(closest_points[0], filtered_start_points[i], closest_points[1], 5, 60, center_rect, hull_rect.height * 0.3))
			{
				finger_points.push_back(filtered_start_points[i]);
			}
		}

		if (finger_points.size() > 0)
		{
			while (finger_points.size() > 5)
			{
				finger_points.pop_back();
			}
			for (int i = 0; i < finger_points.size() - 1; i++)
			{
				if (findPointsDistanceOnX(finger_points[i], finger_points[i + 1]) > hull_rect.height * 0.05 * 1.5)
				{
					filtered_finger_points.push_back(finger_points[i]);
				}

				if (finger_points.size() > 2)
				{
					if (findPointsDistanceOnX(finger_points[0], finger_points[finger_points.size() - 1]) > hull_rect.height * 0.05 * 1.5)
					{
						filtered_finger_points.push_back(finger_points[finger_points.size() - 1]);
					}
				}
				else
				{
					filtered_finger_points.push_back(finger_points[finger_points.size() - 1]);
				}
			}
		}
	}

	cv::polylines(dst, hulls, true, cv::Scalar(0, 255, 255));
	cv::rectangle(dst, hull_rect, cv::Scalar(255, 0, 255));
	cv::circle(dst, center_rect, 5, cv::Scalar(255, 255, 0), 2);
	// drawVectorPoints(frame, filtered_start_points, cv::Scalar(0, 0, 255), true);
	// drawVectorPoints(frame, filtered_far_points, cv::Scalar(0, 0, 255), true);
	drawVectorPoints(dst, filtered_finger_points, cv::Scalar(0, 0, 255), false);
	cv::putText(dst, std::to_string(filtered_finger_points.size()),
		cv::Point(30, 40), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255));
}


void HandDetector::DetectContours(cv::Mat& frame, cv::Mat& result_hsv, cv::Mat& result_masked)
{
	if (frame.empty()) { return; }

	cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
	cv::inRange(frame_hsv, hsv_lower, hsv_upper, result_hsv);
	cv::bitwise_and(result_hsv, mask_f, result_masked);

	cv::medianBlur(result_hsv, result_hsv, 3);
	cv::medianBlur(result_masked, result_masked, 3);

	cv::rectangle(frame, rect_area, cv::Scalar(255, 255, 255));

	cv::findContours(result_masked, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	if (contours.size() <= 0) { return; }

	int biggest_index = -1;
	double biggest_area = 0.0;
	for (int i = 0; i < contours.size(); i++)
	{
		double area_c = cv::contourArea(contours[i]);

		if (area_c < 1e3 || 1e5 < area_c) { continue; }

		if (area_c > biggest_area)
		{
			biggest_area = area_c;
			biggest_index = i;
		}

		cv::drawContours(frame, contours, i, cv::Scalar(0, 255, 0), 1, 8);
	}

	if (biggest_index < 0) { return; }

	cv::convexHull(cv::Mat(contours[biggest_index]), hulls);
	cv::convexHull(cv::Mat(contours[biggest_index]), hull_index);
	hull_rect = cv::boundingRect(cv::Mat(hulls));

	std::vector<cv::Vec4i> defects;
	if (hull_index.size() > 3)
	{
		cv::convexityDefects(cv::Mat(contours[biggest_index]), hull_index, defects);
	}
	else { return; }

	std::vector<cv::Point> start_points;
	std::vector<cv::Point> far_points;

	cv::Point center_rect((hull_rect.tl().x + hull_rect.br().x) / 2,
		(hull_rect.tl().y + hull_rect.br().y) / 2);

	for (int i = 0; i < defects.size(); i++)
	{
		start_points.push_back(contours[biggest_index][defects[i].val[0]]);
		if (findPointsDistance(contours[biggest_index][defects[i].val[2]], center_rect) < hull_rect.height * 0.3)
		{
			far_points.push_back(contours[biggest_index][defects[i].val[2]]);
		}
	}

	std::vector<cv::Point> filtered_start_points = compactOnNeighborhoodMedian(start_points, hull_rect.height * 0.05);
	std::vector<cv::Point> filtered_far_points = compactOnNeighborhoodMedian(far_points, hull_rect.height * 0.05);

	std::vector<cv::Point> filtered_finger_points;
	if (filtered_far_points.size() > 1)
	{
		std::vector<cv::Point> finger_points;
		for (int i = 0; i < filtered_start_points.size(); i++) {
			std::vector<cv::Point> closest_points = findClosestOnX(filtered_far_points, filtered_start_points[i]);

			if (isFinger(closest_points[0], filtered_start_points[i], closest_points[1], 5, 60, center_rect, hull_rect.height * 0.3))
			{
				finger_points.push_back(filtered_start_points[i]);
			}
		}

		if (finger_points.size() > 0)
		{
			while (finger_points.size() > 5)
			{
				finger_points.pop_back();
			}
			for (int i = 0; i < finger_points.size() - 1; i++)
			{
				if (findPointsDistanceOnX(finger_points[i], finger_points[i + 1]) > hull_rect.height * 0.05 * 1.5)
				{
					filtered_finger_points.push_back(finger_points[i]);
				}

				if (finger_points.size() > 2)
				{
					if (findPointsDistanceOnX(finger_points[0], finger_points[finger_points.size() - 1]) > hull_rect.height * 0.05 * 1.5)
					{
						filtered_finger_points.push_back(finger_points[finger_points.size() - 1]);
					}
				}
				else
				{
					filtered_finger_points.push_back(finger_points[finger_points.size() - 1]);
				}
			}
		}
	}

	cv::polylines(frame, hulls, true, cv::Scalar(0, 255, 255));
	cv::rectangle(frame, hull_rect, cv::Scalar(255, 0, 255));
	cv::circle(frame, center_rect, 5, cv::Scalar(255, 255, 0), 2);
	// drawVectorPoints(frame, filtered_start_points, cv::Scalar(0, 0, 255), true);
	// drawVectorPoints(frame, filtered_far_points, cv::Scalar(0, 0, 255), true);
	drawVectorPoints(frame, filtered_finger_points, cv::Scalar(0, 0, 255), false);
	cv::putText(frame, std::to_string(filtered_finger_points.size()),
		        cv::Point(30, 40), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255));
}


double findPointsDistance(cv::Point a, cv::Point b)
{
	cv::Point difference = a - b;
	return sqrt(difference.ddot(difference));
}


std::vector<cv::Point> compactOnNeighborhoodMedian(std::vector<cv::Point> points, double max_neighbor_distance)
{
	std::vector<cv::Point> median_points;

	if (points.size() == 0)
	{
		return median_points;
	}

	if (max_neighbor_distance <= 0)
	{
		return median_points;
	}

	// we start with the first point
	cv::Point reference = points[0];
	cv::Point median = points[0];

	for (int i = 1; i < points.size(); i++)
	{
		if (findPointsDistance(reference, points[i]) > max_neighbor_distance)
		{

			// the point is not in range, we save the median
			median_points.push_back(median);

			// we swap the reference
			reference = points[i];
			median = points[i];
		}
		else
		{
			median = (points[i] + median) / 2;
		}
	}

	// last median
	median_points.push_back(median);

	return median_points;
}


double findPointsDistanceOnX(cv::Point a, cv::Point b)
{
	double to_return = 0.0;

	if (a.x > b.x)
	{
		to_return = static_cast<double>(a.x) - static_cast<double>(b.x);
	}
	else
	{
		to_return = static_cast<double>(b.x) - static_cast<double>(a.x);
	}

	return to_return;
}


std::vector<cv::Point> findClosestOnX(std::vector<cv::Point> points, cv::Point pivot)
{
	std::vector<cv::Point> to_return(2);

	if (points.size() == 0)
		return to_return;

	double distance_x_1 = DBL_MAX;
	double distance_1 = DBL_MAX;
	double distance_x_2 = DBL_MAX;
	double distance_2 = DBL_MAX;
	int index_found = 0;

	for (int i = 0; i < points.size(); i++)
	{
		double distance_x = findPointsDistanceOnX(pivot, points[i]);
		double distance = findPointsDistance(pivot, points[i]);

		if (distance_x < distance_x_1 && distance_x != 0 && distance <= distance_1)
		{
			distance_x_1 = distance_x;
			distance_1 = distance;
			index_found = i;
		}
	}

	to_return[0] = points[index_found];

	for (int i = 0; i < points.size(); i++)
	{
		double distance_x = findPointsDistanceOnX(pivot, points[i]);
		double distance = findPointsDistance(pivot, points[i]);

		if (distance_x < distance_x_2 && distance_x != 0 && distance <= distance_2 && distance_x != distance_x_1)
		{
			distance_x_2 = distance_x;
			distance_2 = distance;
			index_found = i;
		}
	}

	to_return[1] = points[index_found];

	return to_return;
}


double findAngle(cv::Point a, cv::Point b, cv::Point c)
{
	double ab = findPointsDistance(a, b);
	double bc = findPointsDistance(b, c);
	double ac = findPointsDistance(a, c);
	return acos((ab * ab + bc * bc - ac * ac) / (2 * ab * bc)) * 180 / CV_PI;
}


bool isFinger(cv::Point a, cv::Point b, cv::Point c,
	          double limit_angle_inf, double limit_angle_sup,
	          cv::Point palm_center, double min_distance_from_palm)
{
	double angle = findAngle(a, b, c);
	if (angle > limit_angle_sup || angle < limit_angle_inf)
	{
		return false;
	}

	// the finger point should not be under the two far points
	int delta_y_1 = b.y - a.y;
	int delta_y_2 = b.y - c.y;
	if (delta_y_1 > 0 && delta_y_2 > 0)
	{
		return false;
	}

	// the two far points should not be both under the center of the hand
	int delta_y_3 = palm_center.y - a.y;
	int delta_y_4 = palm_center.y - c.y;
	if (delta_y_3 < 0 && delta_y_4 < 0)
	{
		return false;
	}

	double distance_from_palm = findPointsDistance(b, palm_center);
	if (distance_from_palm < min_distance_from_palm)
	{
		return false;
	}

	// this should be the case when no fingers are up
	double distance_from_palm_far_1 = findPointsDistance(a, palm_center);
	double distance_from_palm_far_2 = findPointsDistance(c, palm_center);
	if (distance_from_palm_far_1 < min_distance_from_palm / 4 || distance_from_palm_far_2 < min_distance_from_palm / 4)
	{
		return false;
	}

	return true;
}


void drawVectorPoints(cv::Mat image, std::vector<cv::Point> points, cv::Scalar color, bool with_numbers) {
	for (int i = 0; i < points.size(); i++)
	{
		circle(image, points[i], 5, color, 2, 8);
		if (with_numbers)
		{
			putText(image, std::to_string(i), points[i], cv::FONT_HERSHEY_PLAIN, 3, color);
		}
	}
}
