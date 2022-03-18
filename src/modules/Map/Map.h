#pragma once

#include "../Base/Base.hpp"

using namespace std;

class Map
{

public:
    Map();

    ~Map();

    bool loadAndInit(const std::string &map_name, int pyramid_level_);
    bool getCsmMap();

    static const int MAX_PYRAMID_LEVEL = 3;
    static const int RESOLUTION_LEVEL = 5;

    int pyramid_level;

    cv::Mat mapList[MAX_PYRAMID_LEVEL];
    cv::Mat gradList[MAX_PYRAMID_LEVEL];
    cv::Mat csmList[RESOLUTION_LEVEL];
    cv::Mat icpGradMap;
    cv::Mat rawData;
    cv::Mat data;

    const float gaussSigma[3] = {25.0f ,15.0f ,8.0f};
    // const int gaussSize[3] = {25, 25, 25};
     const int gaussSize[3] = {25, 15, 15};

    // const float gaussSigma[3] = {5.0f, 15.0f, 25.0f};
    // const int gaussSize[3] = {5, 15, 25};

    bool inited;
};