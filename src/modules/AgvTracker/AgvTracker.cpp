#include "AgvTracker.h"
#include "../Base/ImgProcess.h"

#include <sys/time.h>
// #include <dirent.h>
#include <time.h>
using namespace std;

AgvTracker::AgvTracker() : intrinsic(350.0f, 280.0f, 100.0f),
                           agvCenter(320.0f, 280.0f),
                           pyramid_level(3),
                           visualScale(0.4f),
                           withPriPose(true),
                           withEvaluation(false),
                           loseDist(0.0f),
                           degeSpan(0.0f),
                           degeTheta(0.0f),
                           showDemo(true)
{

    //init pose
    //data_mapping: 852.136, 1278.95, 1.0943
    //data_test: 1092.06, 1382.2, 1.55519
    //dege_data/data1: 768.3, 1367.65, 0.2024
    //dege_data/data4: 948.857, 1339.72, 1.5079

    // PrioriPose = Pose(1808.0, 2795.0, -3.1);

    PrioriPose = Pose(1092.06, 1382.2, 1.55519);

    pReader = new Reader();

    pMap = new Map();

    pTracking = new Tracking(pMap);

    pCurFrame = new Frame();

    pCurOdomPose = new Pose;
    pPrevOdomPose = new Pose;
}

AgvTracker::~AgvTracker()
{

    delete pReader;
    delete pMap;

    delete pTracking;

    delete pCurFrame;

    delete pCurOdomPose;
    delete pPrevOdomPose;
}

void AgvTracker::run()
{

    // TruthPose = Pose(3.0859f, -7.0663f, 1.5079f, 0.0f);
    // updateTruthPose();
    // getchar();

    int count = 0;

    int odomIdx = 0;
    int lidarIdx = 0;

    float totalErr = 0.0f;
    float maxErr = 0.0f;
    struct timespec start_time, end_time;
    double total_time;

    vector<double>::iterator iter;
    for (iter = pReader->captureTicks.begin(); iter < pReader->captureTicks.end(); iter++, count++)
    {

        double start = ImgProcess::getTime();

        double tick = *iter;

        string imaName = images_path + to_string(tick) + ".png";
        cout << "cur img name: " << imaName << endl;
        pCurFrame->buildFrame(imaName);

        if (withPriPose)
        {

            *pCurOdomPose = pReader->getReferPose(tick, pReader->odomPoses, odomIdx); //vector no pop ?????

            if (count > 0)
                updatePriPose();
        }

        //tracking

        clock_gettime(CLOCK_REALTIME, &start_time);

        robustTracking(count);

        clock_gettime(CLOCK_REALTIME, &end_time);
        total_time = ((double)end_time.tv_nsec - (double)start_time.tv_nsec) / 1000000.0;
        cout << "-----------------total time :" << total_time << " ms" << endl;

        cout << "count: " << count << endl;
        // cout<<"priori pose: "<<PrioriPose.trans(0)<<", "<<PrioriPose.trans(1)<<", "<<PrioriPose.theta<<endl;
        cout << "visual pose: " << pCurFrame->pose.trans(0) << ", " << pCurFrame->pose.trans(1) << ", " << pCurFrame->pose.theta << endl;

        if (showDemo)
        {
            if (count == 0)
                drawAgvCenter(PrioriPose, cv::Vec3b(255, 255, 255)); //draw init pose

            showDemoImage(pCurFrame->degenerative);
        }

        swap(pPrevOdomPose, pCurOdomPose);

        PrioriPose = pCurFrame->pose;

        double end = ImgProcess::getTime();
        cout << "cost time: " << (end - start) << "s" << endl;
        cout << "fps: " << 1.0 / (end - start) << endl;

        getchar();
    }
    clock_gettime(CLOCK_REALTIME, &end_time);
    cv::imwrite("trajectory.png", demoMapImg);
}

void AgvTracker::initTracker()
{

    if (pyramid_level <= 0)
    {
        cout << "[AgvTracker] pyramid level is invalid: " << pyramid_level << endl;
        exit(-1);
    }

    if (!pMap->loadAndInit(map_name, pyramid_level))
        exit(-1);

    pReader->txt_path = data_path + "timestamp/";
    if (!pReader->readPoseAndTicks())
        exit(-1);

    withPriPose = withPriPose && pReader->withOdomPoses;
    withEvaluation = withEvaluation && pReader->withLidarPoses;

    images_path = data_path + "masks/";
    string firstImgName = images_path + to_string(pReader->captureTicks[0]) + ".png";

    cv::Mat firstImg = cv::imread(firstImgName);

    if (firstImg.empty())
    {
        cout << "[AgvTracker] failed to initilize tracker" << endl;
        exit(-1);
    }

    if (!pCurFrame->init(firstImg.rows, firstImg.cols, pyramid_level))
        exit(-1);

    if (!pTracking->init(firstImg.rows, firstImg.cols, pyramid_level))
        exit(-1);

    // cv::resize(pMap->rawData, demoMapImg, cv::Size(pMap->rawData.cols * visualScale, pMap->rawData.rows * visualScale));
    cv::resize(pMap->rawData, demoImg, cv::Size(pMap->rawData.cols * visualScale, pMap->rawData.rows * visualScale));

    cout << "init pose: " << PrioriPose.trans(0) << ", " << PrioriPose.trans(1) << ", " << PrioriPose.theta << endl;
    cout << "pyramid level: " << pyramid_level << endl;
    cout << "with priori odometry pose: " << (withPriPose ? "yes" : "no") << endl;
    cout << "with lidar evaluation: " << (withEvaluation ? "yes" : "no") << endl;
    cout << "show demo: " << (showDemo ? "yes" : "no") << endl;
}

void AgvTracker::robustTracking(int count)
{

    float priDist = (PrioriPose.trans - pCurFrame->pose.trans).norm();

    if (pCurFrame->lost)
    {

        if (count == 0)
            loseDist = 200.0f;
        else
            loseDist += priDist;

        if (loseDist > 10.0f * intrinsic.scale)
        {
            cout << "[AgvTracker] warning: lost too long: " << loseDist / intrinsic.scale << "m" << endl;
        }

        pCurFrame->pose = PrioriPose;
        degeSpan = 0.0f;

        return;
    }
    struct timespec start_singleframe, end_single_frame, start_degecorrect, end_degecorrect;
    double single_frame_use, dege_corr_use;
    clock_gettime(CLOCK_REALTIME, &start_singleframe);
    //tracking with pytamid or not
    int start_level = ((count == 0 || loseDist > 50.0f)) ? pyramid_level - 1 : 0;

    // pTracking->trackingSingleFrame(pCurFrame, PrioriPose, start_level, 0);
    pTracking->icpTracking(pCurFrame, PrioriPose);

    clock_gettime(CLOCK_REALTIME, &end_single_frame);
    single_frame_use = ((double)end_single_frame.tv_nsec - (double)start_singleframe.tv_nsec) / 1000000.0;
    cout << "-------------single_frame_use:" << single_frame_use << " ms" << endl;

    /**
    if (pCurFrame->degenerative)
    {

        if (count == 0)
        {

            degeSpan = 200.0f;
        }
        else if (loseDist > 100.0f)
        {

            degeSpan += 0.5f * loseDist;
        }
        else
        {

            degeSpan += 0.3f * priDist;
        }

        float theta = pCurFrame->degeTheta + pCurFrame->pose.theta;

        //assert(abs(degeTheta - theta) < 0.3f);
        degeTheta = theta;

        cout << "dege span: " << degeSpan << endl;
        cout << "dege theta: " << degeTheta << endl;
    }
    else
    {

        if (degeSpan < 50.0f)
        {

            degeSpan = 0.0f;
        }
        // else if(pTracking->degenerateCorrect(pCurFrame, degeSpan, degeTheta)){//????
        //     degeSpan = 0.0f;
        //         //getchar();
        // }

        else if (pCurFrame->correctCheck(degeTheta))
        { //????
            // getchar();
            clock_gettime(CLOCK_REALTIME, &start_degecorrect);
            if (pTracking->degenerateCorrect(pCurFrame, degeSpan, degeTheta))
            {

                degeSpan = 0.0f;
                //getchar();
            }
            // pTracking->icpTracking(pCurFrame, PrioriPose);
            clock_gettime(CLOCK_REALTIME, &end_degecorrect);
            dege_corr_use = ((double)end_degecorrect.tv_nsec - (double)start_degecorrect.tv_nsec) / 1000000.0;
            cout << "#############dege_corr_use:" << dege_corr_use << " ms" << endl;
        }
    }
    */
    loseDist = 0.0f;
}

// using imformation of current odom , update the pose as "PriPose"
void AgvTracker::updatePriPose()
{

    // cout<<"update priori pose..."<<endl;

    float cosTheta = cos(pPrevOdomPose->theta);
    float sinTheta = sin(pPrevOdomPose->theta);

    Eigen::Matrix2f inv_R1;
    inv_R1 << cosTheta, sinTheta, -sinTheta, cosTheta;

    Eigen::Vector2f inv_t1 = -inv_R1 * pPrevOdomPose->trans;

    Eigen::Matrix3f inv_T1;
    inv_T1 << inv_R1(0, 0), inv_R1(0, 1), intrinsic.scale * inv_t1(0),
        inv_R1(1, 0), inv_R1(1, 1), intrinsic.scale * inv_t1(1),
        0.0f, 0.0f, 1.0f;

    Eigen::Matrix3f T2;
    T2 << cos(pCurOdomPose->theta), -sin(pCurOdomPose->theta), intrinsic.scale * pCurOdomPose->trans(0),
        sin(pCurOdomPose->theta), cos(pCurOdomPose->theta), intrinsic.scale * pCurOdomPose->trans(1),
        0.0f, 0.0f, 1.0f;

    Eigen::Matrix3f X = Eigen::Matrix3f::Identity(); //AGV to image
    X(0, 0) = -1.0f;
    X(1, 1) = -1.0f;
    X(0, 2) = intrinsic.cx;
    X(1, 2) = intrinsic.cy;

    Eigen::Matrix3f T_prev;
    T_prev << cos(PrioriPose.theta), -sin(PrioriPose.theta), PrioriPose.trans(0),
        sin(PrioriPose.theta), cos(PrioriPose.theta), PrioriPose.trans(1),
        0.0f, 0.0f, 1.0f;

    Eigen::Matrix3f T_pri = T_prev * X * inv_T1 * T2 * X;

    //resize theta
    float theta1 = atan(T_pri(1, 0) / T_pri(0, 0));
    // theta1 = (theta1 > 3.1416f) ? (theta1 - 3.1416f) : theta1;
    // theta1 = (theta1 < -3.1416f) ? (theta1 + 3.1416f) : theta1;

    // float theta2 = (theta1 > 0.0f) ? (theta1 - 3.1416f) : theta1 + 3.1416f;
    // float refDelta = getDeltaTheta(pCurOdomPose->theta, pPrevOdomPose->theta);

    // PrioriPose.theta = (abs(getDeltaTheta(theta1, PrioriPose.theta) - refDelta) <
    //                      abs(getDeltaTheta(theta2, PrioriPose.theta) - refDelta)) ? theta1 : theta2;
    PrioriPose.theta = atan2(T_pri(1, 0), T_pri(0, 0));

    PrioriPose.trans(0) = T_pri(0, 2);
    PrioriPose.trans(1) = T_pri(1, 2);

    //cout<<"raw pose: "<<pCurOdomPose->trans(0)<<" "<<pCurOdomPose->trans(1)<<" "<<pCurOdomPose->theta<<endl;

    //cout<<"priori pose: "<<PrioriPose.trans(0)<<" "<<PrioriPose.trans(1)<<" "<<PrioriPose.theta<<endl;
}
void AgvTracker::updateTruthPose()
{

    Eigen::Matrix3f T_w2a; //world to AGV
    float cosTheta = cos(TruthPose.theta);
    float sinTheta = sin(TruthPose.theta);
    T_w2a << cosTheta, -sinTheta, intrinsic.scale * TruthPose.trans(0),
        sinTheta, cosTheta, intrinsic.scale * TruthPose.trans(1),
        0.0f, 0.0f, 1.0f;

    Eigen::Matrix3f T_a2i = Eigen::Matrix3f::Identity(); //AGV to image
    T_a2i(0, 0) = -1.0f;
    T_a2i(1, 1) = -1.0f;
    T_a2i(0, 2) = intrinsic.cx;
    T_a2i(1, 2) = intrinsic.cy;

    Eigen::Matrix3f T_m2w = T_a2i; //map to world
    T_m2w(0, 2) = (float)pMap->data.rows * visualScale;
    T_m2w(1, 2) = (float)pMap->data.cols * visualScale;

    Eigen::Matrix3f T_m2i = T_m2w * T_w2a * T_a2i;

    //resize theta
    float theta1 = atan(T_m2i(1, 0) / T_m2i(0, 0));
    theta1 = (theta1 > 3.1416f) ? (theta1 - 3.1416f) : theta1;
    theta1 = (theta1 < -3.1416f) ? (theta1 + 3.1416f) : theta1;

    float theta2 = (theta1 > 0.0f) ? (theta1 - 3.1416f) : theta1 + 3.1416f; // theta2 is what????

    TruthPose.theta = (abs(getDeltaTheta(theta1, TruthPose.theta)) <
                       abs(getDeltaTheta(theta2, TruthPose.theta)))
                          ? theta1
                          : theta2;

    TruthPose.trans(0) = T_m2i(0, 2);
    TruthPose.trans(1) = T_m2i(1, 2);

    //cout<<"truth pose: "<<TruthPose.trans(0)<<", "<<TruthPose.trans(1)<<", "<<TruthPose.theta<<endl;
}

void AgvTracker::showDemoImage(bool mark)
{
    // int drawLevel = pyramid_level-1;
    int drawLevel = 0;

    float factor = pow(2, -drawLevel);

    Pose drawPose = pCurFrame->pose;
    drawPose.trans = factor * drawPose.trans;
    cv::Mat frameWindow = ImgProcess::drawFrameToMap(pCurFrame->gaussMaskList[drawLevel], pMap->mapList[drawLevel], drawPose);
    cv::imshow("current frame", frameWindow);
    cv::waitKey(10);

    // cv::imshow("current frame", pCurFrame->gaussMaskList[0]);
    // cv::waitKey(10);

    // if (withEvaluation)
    //     drawAgvCenter(TruthPose, cv::Vec3b(0, 0, 255));

    cv::Vec3b color = (mark) ? cv::Vec3b(255, 0, 0) : cv::Vec3b(0, 255, 255);
    drawAgvCenter(pCurFrame->pose, color);

    cv::namedWindow("trajectory", 0);
    cv::resizeWindow("trajectory", 800, 800);
    cv::imshow("trajectory", demoImg);
    cv::waitKey(10);
}

void AgvTracker::drawAgvCenter(Pose &pose, cv::Vec3b color)
{
    float theta = pose.theta;

    Eigen::Matrix2f Rota;
    Rota << cos(theta), -sin(theta), sin(theta), cos(theta);

    Eigen::Vector2f pixelCenter = Rota * agvCenter + pose.trans;
    demoImg.at<cv::Vec3b>(visualScale * pixelCenter(0), visualScale * pixelCenter(1)) = color;
    cv::circle(demoImg, cv::Point2i(visualScale * pixelCenter(1), visualScale * pixelCenter(0)), 5, cv::Vec3b(0, 255, 255));
}

float AgvTracker::getDeltaTheta(float theta1, float theta2)
{

    float delta = theta1 - theta2;

    delta = (delta > 3.1416f) ? (delta - 6.2832f) : delta;
    delta = (delta < -3.1416f) ? (delta + 6.2832f) : delta;

    return delta;
}
