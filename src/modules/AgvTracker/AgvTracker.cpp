#include "AgvTracker.h"
#include "../Base/ImgProcess.h"

#include <sys/time.h>
// #include <dirent.h>
#include <time.h>
using namespace std;

AgvTracker::AgvTracker() : intrinsic(350.0f, 280.0f, 100.0f),
                           agvCenter(320.0f, 280.0f),
                           pyramid_level(3),
                           visualScale(0.5f),
                           withPriPose(true),
                           withEvaluation(false),
                           loseDist(0.0f),
                           degeSpan(0.0f),
                           degeTheta(0.0f),
                           showDemo(true),
                           pViewer(nullptr)
{

    //init pose
    //data_mapping: 852.136, 1278.95, 1.0943
    //data_test: 1092.06, 1382.2, 1.55519
    //dege_data/data1: 768.3, 1367.65, 0.2024
    //dege_data/data4: 948.857, 1339.72, 1.5079

    PrioriPose = Pose(1808.0, 2795.0, -3.1);

    // PrioriPose = Pose(1092.06, 1382.2, 1.55519);
    // PrioriPose = Pose(948.857, 1339.72, 1.5079);

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

        //evaluation with lidar
        if (withEvaluation)
        {

            TruthPose = pReader->getReferPose(tick, pReader->lidarPoses, lidarIdx);

            // handEyeCalib(count);

            updateTruthPose();

            float err = ((1.0f / intrinsic.scale) * (TruthPose.trans - pCurFrame->pose.trans)).norm(); //error only consider positon not angle.??????

            totalErr += err;
            maxErr = (err > maxErr) ? err : maxErr;

            cout << "current error: " << err << endl;
            cout << "average error: " << totalErr / (count + 1) << endl;
            cout << "max error: " << maxErr << endl;
        }

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
    // cv::imwrite("trajectory.png", demoMapImg);
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

    cv::resize(pMap->rawData, demoMapImg, cv::Size(pMap->rawData.cols * visualScale, pMap->rawData.rows * visualScale));

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
    int start_level = (count == 0 || loseDist > 50.0f) ? pyramid_level - 1 : 0;

    if (count < 10)
        pTracking->icpTracking(pCurFrame, PrioriPose);
    else
        pTracking->trackingSingleFrame(pCurFrame, PrioriPose, start_level, 0);

    clock_gettime(CLOCK_REALTIME, &end_single_frame);
    single_frame_use = ((double)end_single_frame.tv_nsec - (double)start_singleframe.tv_nsec) / 1000000.0;
    cout << "-------------single_frame_use:" << single_frame_use << " ms" << endl;

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
            getchar();
            cout << "degeSpan: " << degeSpan << endl;
            clock_gettime(CLOCK_REALTIME, &start_degecorrect);
            if (pTracking->degenerateCorrect(pCurFrame, degeSpan, degeTheta))
            {

                degeSpan = 0.0f;
                //getchar();
            }
            clock_gettime(CLOCK_REALTIME, &end_degecorrect);
            dege_corr_use = ((double)end_degecorrect.tv_nsec - (double)start_degecorrect.tv_nsec) / 1000000.0;
            cout << "#############dege_corr_use:" << dege_corr_use << " ms" << endl;
        }
    }

    loseDist = 0.0f;
}

void AgvTracker::handEyeCalib(int count)
{

    if (count % 30 != 7)
        return;

    float cosTheta = cos(TruthPose.theta);
    float sinTheta = sin(TruthPose.theta);

    Eigen::Matrix3f calib_curAgvT;
    calib_curAgvT << cosTheta, sinTheta, TruthPose.trans(0),
        -sinTheta, cosTheta, TruthPose.trans(1),
        0.0f, 0.0f, 1.0f;

    cosTheta = cos(pCurFrame->pose.theta);
    sinTheta = sin(pCurFrame->pose.theta);

    Eigen::Matrix3f calib_curImaT;
    calib_curImaT << cosTheta, sinTheta, pCurFrame->pose.trans(0),
        -sinTheta, cosTheta, pCurFrame->pose.trans(1),
        0.0f, 0.0f, 1.0f;

    if (count / 30 != 0)
    {

        Eigen::Matrix3f calib_deltaImaT = calib_prevImaT.inverse() * calib_curImaT;
        Eigen::Matrix3f calib_deltaAgvT = calib_prevAgvT.inverse() * calib_curAgvT;

        Eigen::Matrix2f deltaRota = 0.5f * (calib_deltaImaT.block(0, 0, 2, 2) + calib_deltaAgvT.block(0, 0, 2, 2));
        Eigen::Matrix2f coeffMat = Eigen::Matrix2f::Identity() - deltaRota;

        float maxVal = coeffMat.maxCoeff();
        float minVal = coeffMat.minCoeff();
        if (abs(maxVal) > 0.2f || abs(minVal) > 0.2f)
        {

            deltaImaTrans = calib_deltaImaT.block(0, 2, 2, 1);
            deltaAgvTrans = calib_deltaAgvT.block(0, 2, 2, 1);
            calibTrans = coeffMat.inverse() * (100 * deltaAgvTrans + deltaImaTrans);

            cout << endl
                 << "delta agv rota: " << calib_deltaAgvT.block(0, 0, 2, 2) << endl;
            cout << "delta image rota: " << calib_deltaImaT.block(0, 0, 2, 2) << endl
                 << endl;

            cout << "delta agv trans: " << calib_deltaAgvT.block(0, 2, 2, 1) << endl;
            cout << "delta imge trans: " << calib_deltaImaT.block(0, 2, 2, 1) << endl
                 << endl;

            cout << "A: " << coeffMat << endl;
            cout << "b: " << 100 * deltaAgvTrans + deltaImaTrans << endl
                 << endl;

            cout << "calib params: " << calibTrans << endl
                 << endl;

            cout << "trans1: " << deltaRota * calibTrans + 100 * calib_deltaAgvT.block(0, 2, 2, 1) << endl;
            cout << "trans2: " << calibTrans - calib_deltaImaT.block(0, 2, 2, 1) << endl
                 << endl;

            Eigen::Vector2f priParams(350.0f, 280.0f);
            cout << "evalutation trans1: " << deltaRota * priParams + 100 * calib_deltaAgvT.block(0, 2, 2, 1) << endl;
            cout << "evalutation trans2: " << priParams - calib_deltaImaT.block(0, 2, 2, 1) << endl;

            // getchar();
        }
    }

    calib_prevImaT = calib_curImaT;
    calib_prevAgvT = calib_curAgvT;
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
    T_m2w(0, 2) = (float)pMap->data.rows * 0.4f;
    T_m2w(1, 2) = (float)pMap->data.cols * 0.4f;

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

    cv::Mat overlapImg = ImgProcess::drawFrameToMap(pCurFrame->gaussMaskList[0], pMap->mapList[0], pCurFrame->pose);

    // cv::imshow("current frame", frameWindow);
    // cv::waitKey(10);

    if (withEvaluation)
        drawAgvCenter(TruthPose, cv::Vec3b(0, 0, 255));

    mark = false;
    cv::Vec3b color = (mark) ? cv::Vec3b(255, 0, 0) : cv::Vec3b(0, 255, 255);
    drawAgvCenter(pCurFrame->pose, color);

    pViewer->updateDemo(pCurFrame->pose, pCurFrame->rawMask, overlapImg, demoMapImg);

    // cv::imshow("trajectory", demoImg);
    // cv::waitKey(10);
}

void AgvTracker::drawAgvCenter(Pose &pose, cv::Vec3b color)
{

    float theta = pose.theta;

    Eigen::Matrix2f Rota;
    Rota << cos(theta), -sin(theta), sin(theta), cos(theta);

    Eigen::Vector2f pixelCenter = Rota * agvCenter + pose.trans;

    demoMapImg.at<cv::Vec3b>(visualScale * pixelCenter(0), visualScale * pixelCenter(1)) = color;
}

float AgvTracker::getDeltaTheta(float theta1, float theta2)
{

    float delta = theta1 - theta2;

    delta = (delta > 3.1416f) ? (delta - 6.2832f) : delta;
    delta = (delta < -3.1416f) ? (delta + 6.2832f) : delta;

    return delta;
}
