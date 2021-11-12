//
// Created by surui on 2020/7/1.
//

#include "GeneralSeg.h"

GeneralSeg::GeneralSeg(std::string modelPath)
{
    params.modelPath = modelPath;
    // You can reference your prototxt to fill these field.
    params.batchSize = 1;
    params.resizedHeight = 1024;
    params.resizedWidth = 1024;
    params.inputC = 3;
    params.classNum = 3;

    if (!validateGparams(params)) {
        perror("[ERROR] Check your gparams !\n\n");
    }
    net.load_model(params.modelPath.c_str());
}

void GeneralSeg::init(std::string modelPath)
{
    params.modelPath = modelPath;
    // You can reference your prototxt to fill these field.
    params.batchSize = 1;
    params.resizedHeight = 1024;
    params.resizedWidth = 1024;
    params.inputC = 3;
    params.classNum = 3;

    if (!validateGparams(params)) {
        perror("[ERROR] Check your gparams !\n\n");
    }
    net.load_model(params.modelPath.c_str());
}

bool GeneralSeg::validateGparams(nnie::gParams gparams)
{
    if (gparams.resizedHeight < 1 || gparams.resizedWidth < 1 || gparams.inputC < 1) {
        perror("[ERROR] You have to assign the resize info and channel !\n\n");
        return false;
    }
    if (gparams.batchSize < 1) {
        perror("[ERROR] You have to assign the batch size !\n\n");
        return false;
    }

    if (gparams.classNum < 1) {
        perror("[ERROR] You have to assign the class num !\n\n");
        return false;
    }
    if (gparams.modelPath.empty()) {
        perror("[ERROR] You have to assign the engine path !\n\n");
        return false;
    }
    return true;
}

GeneralSeg::~GeneralSeg() { net.clear(); }

void GeneralSeg::run(std::string imgPath, cv::Mat &clsIdxMask, cv::Mat &colorMask, float threshold)
{
    cv::Mat im = params.inputC == 1 ? cv::imread(imgPath, 0) : cv::imread(imgPath);
    std::cout << imgPath << " : (" << im.rows << ", " << im.cols << ")" << std::endl;
    bool is_none_background_exist = run(im, clsIdxMask, colorMask, threshold);
    std::cout << "None background detected: " << is_none_background_exist << std::endl;
    // const std::string outputImage = imgPath + "_result.png";
    // cv::imwrite(outputImage, colorMask);
}

bool GeneralSeg::run(cv::Mat input_img, cv::Mat &clsIdxMask, cv::Mat &colorMask, float threshold)
{
    if (input_img.rows != params.resizedHeight || input_img.cols != params.resizedWidth)
        cv::resize(input_img, input_img, cv::Size(params.resizedWidth, params.resizedHeight));

    nnie::Mat in;
    nnie::resize_bilinear(input_img, in, params.resizedWidth, params.resizedHeight, params.inputC);

    net.run(in.im);

    nnie::Mat logit;
    net.extract(0, logit);

#ifdef __DEBUG__
    printf("\nTensor h : %d\n", logit.height);
    printf("Tensor w : %d\n", logit.width);
    printf("Tensor c : %d\n", logit.channel);
    printf("Tensor n : %d\n", logit.n);
#endif

    clsIdxMask = cv::Mat::zeros(params.resizedHeight, params.resizedWidth, CV_8UC1);
    colorMask = cv::Mat::zeros(params.resizedHeight, params.resizedWidth, CV_8UC3);

    bool is_none_background_exist = false;
    is_none_background_exist = parseTensor(logit, clsIdxMask, colorMask, threshold);

    free(in.im);

    return is_none_background_exist;
}

bool GeneralSeg::parseTensor(nnie::Mat outTensor, cv::Mat clsIdxMask, cv::Mat &colorMask, float threshold)
{
    bool is_none_backgound_exist = false;
    float *res = outTensor.data;
#pragma omp parallel for
    for (int i = 0; i < params.resizedHeight; ++i) {
        for (int j = 0; j < params.resizedWidth; ++j) {
            float max = 0.0;
            int maxIdx = 0;
            // for (int c = 0; c < params.classNum; ++c) {
            //     float logit = res[j + (i * params.resizedWidth) + c * params.resizedHeight * params.resizedWidth];
            //     //                printf("logit : %f\n", logit);
            //     if (logit > max) {
            //         maxIdx = c;
            //         max = logit;
            //     }
            // }
            // // if any pixel not background
            // if (0 != maxIdx) {
            //     is_none_backgound_exist = true;
            // }
            // //            printf("maxIdx : %f\n", maxIdx);
            std::vector<float> logits(params.classNum);
            for (int c = 0; c < params.classNum; c++) {
                logits[c] = res[j + (i * params.resizedWidth) + c * params.resizedHeight * params.resizedWidth];
            }
            Softmax(logits);
            for (int c = 0; c < params.classNum; c++) {
                if (logits[c] > max & logits[c] > threshold) {
                    maxIdx = c;
                    max = logits[c];
                }
            }
            if (maxIdx != 0) {
                is_none_backgound_exist = true;
#ifdef __DEBUG__
                for (int c = 0; c < params.classNum; c++) {
                    std::cout << c << ": " << logits[c] << " ";
                }
                std::cout << maxIdx;
                std::cout << std::endl;
#endif
            }
            clsIdxMask.at<uchar>(i, j) = maxIdx;
            colorMask.at<cv::Vec3b>(i, j) = colorMap[maxIdx];
        }
    }

    return is_none_backgound_exist;
}

#ifdef __DEBUG__
// Example: General image segmeatation
// ========================= main =================================

int main(int argc, char *argv[])
{
    // if (4 != argc) {
    //     std::cerr << "Usage: $0 model_path threshold img_path" << std::endl;
    //     return -1;
    // }

    std::string ModelPath = "/root/NNIE-lite/data/nnie_model/segmentation/wood_defect2.wk";
    // std::string imgFile = "/root/data/wd/Image_20211022153918349.png";
    // std::string imgFile;
    float threshold = 0.999;
    std::cout << ModelPath << std::endl;
    std::cout << threshold << std::endl;

    GeneralSeg enet;
    enet.init(ModelPath);
    cv::Mat clsMask;
    cv::Mat colorMask;
    std::vector<cv::String> filenames;
    cv::String folder = "/root/data/wd";
    cv::String postfix = "_result.png";
    cv::glob(folder, filenames);
    for (size_t i = 0; i < filenames.size(); i++) {
        std::cout << filenames[i] << std::endl;
        cv::Mat src = cv::imread(filenames[i]);
        if (!src.data) {
            std::cerr << "Error Reading file " << filenames[i] << std::endl;
        }
        enet.run(src, clsMask, colorMask, threshold);
        cv::imwrite(filenames[i] + postfix, colorMask);
    }
}
#endif
