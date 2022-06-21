#include "yolo_post.h"
void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].confidence;

    while (i <= j) {
        while (objects[i].confidence > p) i++;
        while (objects[j].confidence < p) j--;

        if (i <= j) {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (right > i) qsort_descent_inplace(objects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void yolov5_generate_proposals(Tensor feature, const std::vector<cv::Size2f>& anchor, int stride, float prob_threshold,
                               std::vector<Object>& objects)
{
    int num_class = feature.height - 5;
    int anchor_nums = feature.channel;
    int map_w = sqrt(feature.width);

    for (int anchor_index = 0; anchor_index < anchor_nums; ++anchor_index) {
        int anchor_stride = anchor_index * feature.width * feature.height;
        float anchor_w = anchor[anchor_index].width;
        float anchor_h = anchor[anchor_index].height;

        for (int num_grid = 0; num_grid < feature.width; ++num_grid) {
            int y = num_grid / map_w;
            int x = num_grid % map_w;

            float confidence = feature.data[anchor_stride + 4 * feature.width + num_grid];
            confidence = sigmoid(confidence);

            if (confidence > prob_threshold) {
                for (int class_index = 5; class_index < feature.height; ++class_index) {
                    float class_confidence = feature.data[anchor_stride + class_index * feature.width + num_grid];
                    class_confidence = sigmoid(class_confidence) * confidence;

                    if (class_confidence > prob_threshold) {
                        // printf("confidence %f\n", class_confidence);
                        float dx = feature.data[anchor_stride + 0 * feature.width + num_grid];
                        float dy = feature.data[anchor_stride + 1 * feature.width + num_grid];
                        float dw = feature.data[anchor_stride + 2 * feature.width + num_grid];
                        float dh = feature.data[anchor_stride + 3 * feature.width + num_grid];

                        dx = sigmoid(dx);
                        dy = sigmoid(dy);
                        dw = sigmoid(dw);
                        dh = sigmoid(dh);

                        dx = (dx * 2.0f - 0.5f + (float)x) * stride;
                        dy = (dy * 2.0f - 0.5f + (float)y) * stride;
                        dw = (dw * 2.0f) * (dw * 2.0f) * anchor_w;
                        dh = (dh * 2.0f) * (dh * 2.0f) * anchor_h;

                        float x0 = dx - (dw - 1.0f) * 0.5f;
                        float x1 = dx + (dw - 1.0f) * 0.5f;
                        float y0 = dy - (dh - 1.0f) * 0.5f;
                        float y1 = dy + (dh - 1.0f) * 0.5f;

                        Object obj;
                        obj.bbox.x = x0;
                        obj.bbox.y = y0;
                        obj.bbox.width = x1 - x0;
                        obj.bbox.height = y1 - y0;
                        obj.class_label = class_index - 5;
                        obj.confidence = class_confidence;
                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

void yolo_nms(std::vector<Object>& objects, std::vector<size_t>& picked, float nms_confidence)
{
    picked.clear();
    const size_t n = objects.size();

    struct timeval tv1;
    struct timeval tv2;
    long t1, t2, time_run;

#ifdef __DEBUG__
    gettimeofday(&tv1, NULL);
#endif
    std::vector<float> area(n);
    for (size_t i = 0; i < n; ++i) {
        area[i] = objects[i].bbox.area();
    }

    for (size_t i = 0; i < n; ++i) {
        const Object& object_a = objects[i];
        int keep = 1;

        for (size_t j = 0; j < picked.size(); ++j) {
            const Object& object_b = objects[picked[j]];
            // if (object_b.class_label != object_a.class_label)  //类别不一致没有必要做nms
            //     continue;
            float inter_area = intersection_area(object_a, object_b);
            float union_area = area[i] + area[picked[j]] - inter_area;

            if (inter_area / union_area > nms_confidence) keep = 0;
        }
        if (keep) picked.push_back(i);
    }

#ifdef __DEBUG__
    gettimeofday(&tv2, NULL);
    t1 = tv2.tv_sec - tv1.tv_sec;
    t2 = tv2.tv_usec - tv1.tv_usec;
    time_run = (long)(t1 * 1000 + t2 / 1000);
    printf("get area time : %dms\n", time_run);
#endif
}

void draw_objects(cv::Mat& img, std::vector<Object>& objects)
{
    for (size_t i = 0; i < objects.size(); ++i) {
        const Object& object = objects[i];

        // sfprintf(stderr, "%d = %f at %.2f %.2f %.2f %.2f\n", object.class_label, object.confidence, object.bbox.x,
        // object.bbox.y, object.bbox.width, object.bbox.height);
        cv::rectangle(img, object.bbox, cv::Scalar(0, 0, 255), 2);
        // char text[256];
        // sprintf(text, "%s %.1f", class_names[object.class_label], object.confidence);
        // cv::putText(img, text, object.bbox.tl(), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}