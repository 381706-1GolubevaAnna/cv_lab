#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <vector>
#include <math.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;


//grayscaleImage

Mat grayscaleImage(Mat image) {

	Mat grayImage;

	cvtColor(image, grayImage, COLOR_BGR2GRAY);
	imwrite("res/gray.jpg", grayImage);

	namedWindow("image", WINDOW_AUTOSIZE);
	namedWindow("Gray image", WINDOW_AUTOSIZE);

	imshow("image", image);
	imshow("Gray image", grayImage);
	return grayImage;
}

//Increase contrast

void imageContrast(Mat image) {

	Mat res, gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);

	equalizeHist(gray, res);

	imwrite("res/histogramEqualized.jpg", res);

	namedWindow("Image", WINDOW_AUTOSIZE);
	namedWindow("Histogram Equalized Image", WINDOW_AUTOSIZE);

	imshow("Image", image);
	imshow("Histogram Equalized Image", res);
}

//CANNY

void canny(Mat image) {

	Mat canny;
	Mat grayImage, detected_edges;


	cvtColor(image, grayImage, COLOR_BGR2GRAY);

	blur(grayImage, detected_edges, Size(3, 3));

	Canny(detected_edges, canny, 100, 200, 3);

	imwrite("res/imageCanny.jpg", grayImage);

	namedWindow("Image");
	imshow("Image", image);

	namedWindow("Canny");
	imshow("Canny", canny);
}



//distance transform and corner points
Mat distTransform(Mat image, Mat Edges) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	vector<Point2f> cornerPoints;

	goodFeaturesToTrack(gray, cornerPoints, 100, 0.2, 2);

	Scalar color(255, 255, 100, 255);
	Mat distTransformImage = image.clone();
	Mat edgeWithCorners = Edges.clone();
        double minThreshold = 40.0, maxThreshold = 255.0;

	for (auto i = cornerPoints.begin(); i != cornerPoints.end(); i++) {
		circle(edgeWithCorners, *i, 7, color);
	}

	bitwise_not(edgeWithCorners, edgeWithCorners);
	cvtColor(edgeWithCorners, edgeWithCorners, COLOR_BGR2GRAY);

	distanceTransform(edgeWithCorners, distTransformImage, DIST_L2, 3);
	normalize(distTransformImage, distTransformImage, 0, 1., NORM_MINMAX);

	namedWindow("distance transform Image", WINDOW_AUTOSIZE);
	imshow("distance transform Image", distTransformImage);
	return distTransformImage;
}


//filter


int funcMinMax(int val, int max) {
	int min = 0;
	if (val < min) return min;
	if (val > max) return max;

	return val;
}

Point pixel(Mat grayImage, int rad, int x, int y) {
	int size = 2 * rad + 1;
	pair<int, Point>* around = new pair<int, Point>[size * size];

	for (int l = -rad; l < rad + 1; l++) {
		for (int k = -rad; k < rad + 1; k++) {
			int xd = funcMinMax(x + k, grayImage.cols - 1);
			int yd = funcMinMax(y + l, grayImage.rows - 1);

			around[(l + rad) * size + (k + rad)] = make_pair(static_cast<int>(grayImage.at<uchar>(yd, xd)), Point(xd, yd));
		}
	}

	sort(around, around + size * size, [](pair<int, Point> a, pair<int, Point> b)->bool {return a.first < b.first;});
	Point result = (around[(size * size) / 2]).second;
	delete[]around;

	return result;
}

void IntegralImage(Mat image, Mat edges, double k) {
	
	Mat distImage = distTransform(image, edges);
	Mat grayImage = grayscaleImage(image);

	Mat integralImage, res = image.clone();
	integral(image, integralImage);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			int rad = min(static_cast<int>(k * distImage.at<uchar>(Point(i, j))), 4);
			Point p1(j - rad, i - rad), p2(funcMinMax(j + rad, image.cols - 1), i - rad),
						p3(funcMinMax(j + rad, image.cols - 1), funcMinMax(i + rad, image.rows - 1)),
						p4(funcMinMax(j - rad, funcMinMax(i + rad, image.rows - 1)));

			int temp = integralImage.at<int>(p1) + integralImage.at<int>(p3) - integralImage.at<int>(p2) - integralImage.at<int>(p4);
			float t = pow((2 * rad + 1), 2);

			res.at<uchar>(Point(i, j)) = static_cast<uchar>(float(temp) / t);
		}
	}

	imshow("Filter with integral image", res);
	waitKey();
}


//corner points in the image, harris

void harris(Mat image) {

	Mat res, src;
	int thresh = 200;

	cvtColor(image, src, COLOR_BGR2GRAY);

	Mat dst, dst_norm;
	dst = Mat::zeros(src.size(), CV_32FC1);

	cornerHarris(src, dst, 2, 3, 0.05);

	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, res);


	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				circle(res, Point(i, j), 2, Scalar(0), 2, 8, 0);
			}
		}
	}

	namedWindow("Image");
	imshow("Image", src);

	namedWindow("Corners");
	imshow("Corners", res);
}



void Filter(Mat image, Mat edges, double k) {

	Mat distImage = distTransform(image, edges);
	Mat grayImage = grayscaleImage(image);
	float averageRadius = 0.0f;
	Mat res = image.clone();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int rad = min(static_cast<int>(k * distImage.at<uchar>(Point(i, j))), 5);
			Point p = pixel(grayImage, rad, i, j);
			res.at<uchar>(Point(i, j)) = image.at<uchar>(Point(p.x, p.y));
			averageRadius += rad;
		}
	}

	namedWindow("Image", WINDOW_AUTOSIZE);
	imshow("Image", image);

	namedWindow("Filter", WINDOW_AUTOSIZE);
	imshow("Filter", res);
	waitKey();
}

int main()
{
	Mat image = imread("res/image.jpg", IMREAD_COLOR);

	Mat gray = grayscaleImage(image);

	imageContrast(image);

	Mat imageCanny = imread("res/histogramEqualized.jpg", IMREAD_COLOR);

	canny(imageCanny);

	Mat imageHarris = imread("res/image.jpg", IMREAD_COLOR);

	harris(imageHarris);
	Mat dist = distTransform(image, imageCanny);

	Filter(image, imageCanny, 5);
	IntegralImage(image, imageCanny, 5);


	waitKey(0);
	return 0;
}

