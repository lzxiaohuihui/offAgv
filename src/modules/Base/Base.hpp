#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/core/utility.hpp> //opencv3
// #include <opencv2/imgproc/types_c.h> //opencv4

#include <fstream>
#include<ctime> 
#include<math.h>
#include<vector>
#include <map>
#include<sys/time.h>

#include <iostream>
#include <string>
#include <assert.h>

#include <Eigen/Core>
#include <Eigen/Eigen>

#define EQUAL(a,b) (abs(a-b) < 0.00001)

using namespace std;


struct Pose{

    Pose(float tx = 0.0f, float ty = 0.0f, 
        float theta_ = 0.0f, 
        double tick_ = 0.0):
        trans(Eigen::Vector2f(tx, ty)),
        theta(theta_),
        tick(tick_){

    }

    Eigen::Vector2f trans;
    float theta;

    double tick;

};

struct LaneLine{

    LaneLine(Eigen::Vector3f line = Eigen::Vector3f::Zero(), float len = 0.0f){

        length = len;
        data = line;
        theta = atan(line(1)/line(0));

    }

    Eigen::Vector3f data;

    float theta;
    float length;
};

struct Float2{

	float x;
	float y;
};

struct Intr{

    Intr(float cx_ = 0.0f, float cy_ = 0.0f, float scale_ = 1.0f):
        cx(cx_), cy(cy_), scale(scale_){

    }

    float cx;
    float cy;
    float scale;
};

