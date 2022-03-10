
#include "../Tracking/Tracking.h"
#include "../Reader/Reader.h"
#include "../Viewer/Viewer.h"

using namespace std;

class AgvTracker{

public:

    AgvTracker();

    ~AgvTracker();

    void run();

    void initTracker();

    Map* pMap;

    Viewer* pViewer;

    string map_name;

    string data_path;

private:


    void robustTracking(int count);

    void handEyeCalib(int count);

    void updatePriPose();

    void updateTruthPose();

    void showDemoImage(bool mark);

    void drawAgvCenter(Pose& pose, cv::Vec3b color);

    float getDeltaTheta(float theta1, float theta2);


    Reader* pReader;


    Tracking* pTracking;

    Frame* pCurFrame;

    Pose* pCurOdomPose;
    Pose* pPrevOdomPose;

    Pose TruthPose;
    Pose PrioriPose;

    Intr intrinsic; 
    Eigen::Vector2f agvCenter;


    string images_path;

    int pyramid_level;

    float loseDist;
    float degeSpan;
    float degeTheta;

    bool withPriPose;

    bool withEvaluation;

    float visualScale;

    cv::Mat demoMapImg;
    bool showDemo;

    //calib
    Eigen::Matrix3f calib_prevImaT;
    Eigen::Matrix3f calib_prevAgvT;

    Eigen::Vector2f calibTrans;

    Eigen::Vector2f deltaImaTrans;
    Eigen::Vector2f deltaAgvTrans;
    

};