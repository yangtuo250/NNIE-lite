/*
 * @*: ********************************************************************
 * @Copyright: Copyright (c) 2021 ChaoFeng workteam. All rights reserved.
 * @Description:
 * @Version: 1.0
 * @Autor: mahxn0
 * @Date: 2021-12-12 18:03:29
 * @LastEditors: mahxn0
 * @LastEditTime: 2021-12-12 18:29:34
 * @Begin: ********************************************************************
 * @Modify:
 * @End**: ********************************************************************
 */
#ifndef __YOLO_POST_H__
#define __YOLO_POST_H__

#include <sys/time.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "Tensor.h"

static const char* class_names[] = {"corner", "edge", "light", "dark"};
struct Object {
    cv::Rect_<float> bbox;
    int class_label;
    float confidence;
};

inline float sigmoid(float x) { return (1.0f / ((float)exp((double)(-x)) + 1.0f)); }

inline float intersection_area(const Object& a, const Object& b)
{
    float axmin = a.bbox.x;
    float axmax = a.bbox.x + a.bbox.width;
    float aymin = a.bbox.y;
    float aymax = a.bbox.y + a.bbox.height;
    float bxmin = b.bbox.x;
    float bxmax = b.bbox.x + b.bbox.width;
    float bymin = b.bbox.y;
    float bymax = b.bbox.y + b.bbox.height;

    if (axmin > bxmax || axmax < bxmin || aymin > bymax || aymax < bymin) {
        return 0.f;  // no intersection
    }

    // cv::Rect_<float> inter = a.bbox & b.bbox;
    // return inter.area();
    float inter_width = std::min(axmax, bxmax) - std::max(axmin, bxmin);
    float inter_height = std::min(aymax, bymax) - std::max(aymin, bymin);
    return inter_height * inter_width;
}

void yolo_nms(std::vector<Object>& objects, std::vector<size_t>& picked, float nms_confidence);
void yolov5_generate_proposals(Tensor feature, const std::vector<cv::Size2f>& anchor, int stride, float prob_threshold,
                               std::vector<Object>& objects);
void qsort_descent_inplace(std::vector<Object>& objects);
void draw_objects(cv::Mat& img, std::vector<Object>& objects);

#endif