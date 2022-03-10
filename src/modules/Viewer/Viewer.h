#include <pangolin/pangolin.h>

#include "../Base/Base.hpp"


class Viewer {
   public:

    Viewer();

    void run();

    void updateDemo(Pose& pose, cv::Mat& mask, cv::Mat& overlap, cv::Mat& trajMap);

   private:

    cv::Mat demoMap;
    
    Pose curPose;

    cv::Mat overlapMask;
    cv::Mat curMask;

    Eigen::Vector2f agvCenter;

    const int frameRows;
    const int frameCols;

    const int mapRows;
    const int mapCols;

    Eigen::Matrix4f T_SE3;

    bool endSwitch;

    std::mutex dataMutex;
};

