#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <vector>

#include "Tensor.h"
#include "ins_nnie_interface.h"
#include "opencv2/opencv.hpp"
#include "util.h"
#include "yolo_post.h"
/******************************************************************************
 * function : show usage
 ******************************************************************************/
typedef unsigned char U_CHAR;

void yolov5DetectDemo(NNIE &yolov5, cv::Mat &inference_img, std::vector<Object> &objects, float conf_threshold,
                      float nms_threshold)
{
    struct timeval tv1;
    struct timeval tv2;
    long t1, t2, time_run;

#ifdef __DEBUG__
    gettimeofday(&tv1, NULL);
#endif

    yolov5.run(inference_img);

#ifdef __DEBUG__
    gettimeofday(&tv2, NULL);
    t1 = tv2.tv_sec - tv1.tv_sec;
    t2 = tv2.tv_usec - tv1.tv_usec;
    time_run = (long)(t1 * 1000 + t2 / 1000);
    printf("yolov5 NNIE inference time : %dms\n", time_run);

    gettimeofday(&tv1, NULL);
#endif

    Tensor output0 = yolov5.getOutputTensor(0);
    Tensor output1 = yolov5.getOutputTensor(1);
    Tensor output2 = yolov5.getOutputTensor(2);

    // float src_img_w = (float)src_img.cols;
    // float src_img_h = (float)src_img.rows;
    float inference_img_w = (float)inference_img.cols;
    float inference_img_h = (float)inference_img.rows;

    std::vector<Object> proposals;
    std::vector<size_t> picked;

    const std::vector<std::vector<cv::Size2f>> anchors = {{{4.0f, 3.0f}, {8.0f, 7.0f}, {7.0f, 21.0f}},
                                                          {{13.0f, 11.0f}, {40.0f, 5.0f}, {25.0f, 27.0f}}};

    yolov5_generate_proposals(output0, anchors[0], 8, conf_threshold, proposals);

    yolov5_generate_proposals(output1, anchors[1], 16, conf_threshold, proposals);

    // yolov5_generate_proposals(output2, anchors[0], 32, conf_threshold, proposals);

    qsort_descent_inplace(proposals);
#ifdef __DEBUG__
    printf("get %d\n", proposals.size());
    // for (size_t i = 0; i < proposals.size(); ++i) printf("sort_confidence: %f\n", proposals[i].confidence);
#endif

    yolo_nms(proposals, picked, nms_threshold);
#ifdef __DEBUG__
    gettimeofday(&tv2, NULL);
    std::cout << "picked: " << picked.size() << std::endl;
#endif
    // std::vector<Object> objects(picked.size());
    objects.clear();
    objects.resize(picked.size());

    // float scale_w = src_img_w / inference_img_w;
    // float scale_h = src_img_h / inference_img_h;

    for (size_t i = 0; i < picked.size(); ++i) {
        objects[i] = proposals[picked[i]];

        // float x0 = objects[i].bbox.x * scale_w;
        // float x1 = (objects[i].bbox.x + objects[i].bbox.width) * scale_w;
        // float y0 = objects[i].bbox.y * scale_h;
        // float y1 = (objects[i].bbox.y + objects[i].bbox.height) * scale_h;
        float x0 = objects[i].bbox.x;
        float x1 = (objects[i].bbox.x + objects[i].bbox.width);
        float y0 = objects[i].bbox.y;
        float y1 = (objects[i].bbox.y + objects[i].bbox.height);

        // x0 = std::min(std::max(x0, 0.f), src_img_w - 1.0f);
        // x1 = std::min(std::max(x1, 0.f), src_img_w - 1.0f);
        // y0 = std::min(std::max(y0, 0.f), src_img_h - 1.0f);
        // y1 = std::min(std::max(y1, 0.f), src_img_h - 1.0f);
        x0 = std::min(std::max(x0, 0.f), inference_img_w - 1.0f);
        x1 = std::min(std::max(x1, 0.f), inference_img_w - 1.0f);
        y0 = std::min(std::max(y0, 0.f), inference_img_h - 1.0f);
        y1 = std::min(std::max(y1, 0.f), inference_img_h - 1.0f);

        objects[i].bbox.x = x0;
        objects[i].bbox.y = y0;
        objects[i].bbox.width = x1 - x0;
        objects[i].bbox.height = y1 - y0;
    }

    std::cout << "objects: " << objects.size() << std::endl;
    std::cout << "drawing" << std::endl;
    draw_objects(inference_img, objects);

#ifdef __DEBUG__
    t1 = tv2.tv_sec - tv1.tv_sec;
    t2 = tv2.tv_usec - tv1.tv_usec;
    time_run = (long)(t1 * 1000 + t2 / 1000);
    printf("yolov5 postProcess : %dms\n", time_run);
#endif
}

#ifdef __DEBUG__
/******************************************************************************
 * function
 ******************************************************************************/
int main(int argc, char *argv[])
{
    const char *model_path = "/root/NNIE-lite/data/nnie_model/detection/tile_yolo.wk";

    cv::Mat orig_img, img;
    std::vector<Object> objects;

    NNIE yolov5;
    printf("begin to initiate the model\n");
    yolov5.init(model_path, 1024, 1024);

    std::vector<cv::String> filenames;
    cv::String folder = "/root/data/tile";
    cv::String postfix = "_result.png";
    cv::glob(folder, filenames);
    for (size_t i = 0; i < filenames.size(); i++) {
        std::cout << filenames[i] << std::endl;
        orig_img = cv::imread(filenames[i]);
        if (orig_img.empty()) {
            std::cerr << "Error Reading file " << filenames[i] << std::endl;
        }
        resize(orig_img, img, cv::Size(1024, 1024));
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        printf("yolov5 start\n");
        yolov5DetectDemo(yolov5, img, objects, 0.95, 0.01);
        printf("yolov5 finish\n");
        cv::imwrite(filenames[i] + postfix, img);
    }
    yolov5.finish();

    return 0;
}
#endif
