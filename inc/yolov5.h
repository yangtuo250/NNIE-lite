#ifndef __YOLOV5_H__
#define __YOLOV5_H__

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

void yolov5Detect(NNIE &yolov5, cv::Mat &inference_img, std::vector<Object> &objects, float conf_threshold = 0.8,
                      float nms_threshold=0.01);

#endif