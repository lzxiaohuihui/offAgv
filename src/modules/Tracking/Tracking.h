
#include "../Map/Map.h"
#include "../Frame/Frame.h"

// 降采样参数
#define REF_DOWN_SAMPLE_RATE 3
#define NEW_DOWN_SAMPLE_RATE 3
// ICP迭代阈值
#define ITER_MAX 5
#define ITER_THRESH 0.01

#define NORMAL_THRESH 0.1

using namespace std;

class Tracking
{

public:
    Tracking(Map *pMap_);

    ~Tracking();

    bool init(int frameRows, int frameCols, int pyramid_level_);

    bool trackingSingleFrame(Frame *pFrame, Pose &PriPose, int startLevel = 0, int endLevel = 0);

    bool degenerateCorrect(Frame *pFrame, float degeSpan, float degeTheta);

    bool icpTracking(Frame *pFrame, Pose &PriPose);

    bool csmTracking(Frame *pFrame, Pose &PriPose);

    Map *pMap;
    Frame *curFrame;

    bool inited;

private:
    bool findDataAssociation(cv::Mat &ICP_ref_pts, cv::Mat &ICP_new_pts, vector<cv::Point> &exist_association_pts, vector<cv::Point> &nearest_pts, int knn);
    bool downSample(vector<cv::Point> &ref_pt_vec, vector<cv::Point> &new_pt_vec, cv::Mat &ICP_ref_pts, cv::Mat &ICP_new_pts);
    bool findDataAssociation(cv::Mat &ICP_ref_pts, cv::Mat &ICP_new_pts);

    bool icpTrackingWithG_N(vector<cv::Point> &ref_pt_vec, vector<cv::Point> &new_pt_vec);

    bool icpTrackingWithKdtree(vector<cv::Point> &ref_pt_vec, vector<cv::Point> &new_pt_vec);

    bool getLocalMap(const cv::Mat &temp_frame, Eigen::Vector3f params, vector<cv::Point2i> &local_map, vector<cv::Point2i> &icp_frame);

    void getNoEmptyPixels(const cv::Mat &frame);

    float calcResidualError(Eigen::Vector3f params, int level);

    void LM_Optimate(int level, float &refErr);

    void buildNormEquation(int level);

    void pose2Params(int level);

    void params2Pose(int level);

    //degenerate correct
    void getFrameCorrectBuff(cv::Mat &mask, float refTheta);

    float calcCorrectError(Pose &pose, cv::Mat &map);

    void getCandidatePose(Frame *pFrame, float degeSpan, float degeTheta);

    float calcBidirectError(Frame *pFrame, int level);

    float getDeltaTheta(float theta1, float theta2);

    float *pFrame_xyi;
    float *pMap_xy;
    float *pMapGrad_xy;
    float *pRp_xy;
    float *pRes;

    Eigen::Vector3f smoResVec;

    const int MAX_PYRAMID_LEVEL = 3;

    const int MIN_NOEMPTY_PIXEL = 30;

    int OPTIMIZE_BUFFER_SIZE;

    int pyramid_level;

    int noEmptyPixels;
    int validPixels;

    int frameRows;
    int frameCols;

    const int maxIterNum;
    cv::flann::Index My_Kdtree;
    Eigen::Vector3f curParams;

    Eigen::Matrix3f JtJ;
    Eigen::Vector3f Jtr;

    Pose curPose;
    Pose constPose;

    float smoothWeight;
    const float thetaWeight;

    //degenerate correct
    float *pFrameCorrect_xy;

    int searchLevel;
    int optimateLevel;
    int evaluLevel;

    int corPixelCount;

    vector<Pose> candidatePoses;
    vector<float> candidateSpans;

    bool firstOpt;
};