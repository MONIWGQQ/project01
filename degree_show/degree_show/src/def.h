#pragma once

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <vector>

using namespace std;
//using namespace cv;
struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
};

typedef struct PointInfo
{
	cv::Point pt;
	float score;
} PointInfo;

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	PointInfo kpt1;
	PointInfo kpt2;
	PointInfo kpt3;
	PointInfo kpt4;
	PointInfo kpt5;
	float calculateQuadrilateralArea() {
    // 计算两个三角形的面积
		float area1 = calculateTriangleArea(kpt1.pt, kpt2.pt, kpt4.pt);
		float area2 = calculateTriangleArea(kpt1.pt, kpt4.pt, kpt5.pt);

		// 返回四边形的面积
		return area1 + area2;
	}
private:
	float calculateTriangleArea(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3) {
		// 使用海伦公式计算三角形的面积
		float a = distance(p1, p2);
		float b = distance(p2, p3);
		float c = distance(p3, p1);
		float s = (a + b + c) / 2;
		return sqrt(s * (s - a) * (s - b) * (s - c));
	}

	float distance(const cv::Point& p1, const cv::Point& p2) {
		// 计算两点之间的距离
		float dx = p1.x - p2.x;
		float dy = p1.y - p2.y;
		return sqrt(dx * dx + dy * dy);
	}
} BoxInfo;

typedef struct BoxInfo_Hand
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label_id;
} BoxInfo_Hand;
