// #pragma once

#ifndef LIB_IMGPROCESS
#define LIB_IMGPROCESS
#include"Base.hpp"

using namespace std;

class ImgProcess{

public:

    // ImgProcess();

    // ~ImgProcess();


    static bool computeGradientDiff(const cv::Mat& img, cv::Mat& gradeImg);

    static bool imageResizeHalf(const cv::Mat& src, cv::Mat& dst);

    static cv::Mat drawFrameToMap(const cv::Mat& frame, const cv::Mat& map, Pose& pose);

    static bool poseSwitchPyramid(const int srcLevel, const Pose& srcPose, int dstLevel, Pose& dstPose);
    
    static double getTime();

};


#endif
