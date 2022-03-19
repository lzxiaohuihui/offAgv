
#include "Tracking.h"

#include "../Base/ImgProcess.h"

#include <time.h>

using namespace std;

bool Tracking::degenerateCorrect(Frame *pFrame, float degeSpan, float degeTheta)
{

    cout << "[DegeCorrect] degeneration correct..." << endl;

    //float refTheta = degeTheta - pFrame->pose.theta;

    // refTheta is the frame dege theta...
    float refTheta = degeTheta - pFrame->pose.theta;

    struct timespec start_dege_prepare, end_dege_prepare;
    double prepare_dege_corr;
    //clock_gettime(CLOCK_REALTIME,&start_dege_prepare);
    getFrameCorrectBuff(pFrame->maskList[searchLevel], refTheta);

    // corPixelCount is the number of non dege theta pixel
    if (corPixelCount < 30)
    {
        cout << "[DegeCorrect] error: non-empty pixel count too small: " << corPixelCount << endl;
        return false;
    }

    getCandidatePose(pFrame, degeSpan, degeTheta);

    if (candidatePoses.size() == 0)
    {
        cout << "[DegeCorrect] error: can not find any suitable pose" << endl;
        return false;
    }

    //clock_gettime(CLOCK_REALTIME,&end_dege_prepare);
    //prepare_dege_corr = ((double)end_dege_prepare.tv_nsec - (double)start_dege_prepare.tv_nsec)/1000000.0;
    //cout<<"-----------------prepare time0 :"<<prepare_dege_corr<<" ms"<<endl;

    assert(candidateSpans.size() == candidatePoses.size());
    optimateLevel = (candidateSpans.size() > 3) ? searchLevel : (searchLevel - 1); //?????

    Pose rawPose = pFrame->pose;

    float minScore = -1.0f;
    Pose bestPose;

    clock_gettime(CLOCK_REALTIME, &start_dege_prepare);
    for (int i = 0; i < candidatePoses.size(); i++)
    {
        // if (!trackingSingleFrame(pFrame, candidatePoses[i], searchLevel, optimateLevel)) //?????is same as orbslam relocalization?
        //     continue;
        if (!trackingSingleFrame(pFrame, candidatePoses[i], 0, 0))
        {
            continue;
        }

        float error = calcBidirectError(pFrame, evaluLevel); //????? why dont calc as calcCorrectError
        // float score = error;
        float score = error * (0.9f * abs(candidateSpans[i]) + 0.1f);

        if (minScore < 0.0f || score < minScore)
        {

            minScore = score;
            bestPose = pFrame->pose;
            cout << "[after ]tracking single frame, the PriPose is: " << bestPose.trans(0) << "," << bestPose.trans(1) << "," << bestPose.theta << endl;
        }

        cout << "bidirect error: " << error << "  span: " << candidateSpans[i] << "  score: " << score << endl;

        // cv::Mat frameWindow = ImgProcess::drawFrameToMap(pFrame->gaussMaskList[0], pMap->mapList[0], pFrame->pose);
        // cv::imshow("current correct pose", frameWindow);
        // cv::waitKey(100);
        // getchar();
    }
    clock_gettime(CLOCK_REALTIME, &end_dege_prepare);
    prepare_dege_corr = ((double)end_dege_prepare.tv_nsec - (double)start_dege_prepare.tv_nsec) / 1000000.0;
    cout << "-----------------prepare time1 :" << prepare_dege_corr << " ms" << endl;
    prepare_dege_corr = ((double)end_dege_prepare.tv_nsec - (double)start_dege_prepare.tv_nsec) / 1000000.0;
    if (minScore < 0.0f)
    {

        cout << "[DegeCorrect] error: tracking failed" << endl;
        pFrame->pose = rawPose;
        return false;
    }

    if ((bestPose.trans - rawPose.trans).norm() < 5.0f && abs(bestPose.theta - rawPose.theta) < 0.2f)
    {

        cout << "[DegeCorrect] no correction required" << endl;
        pFrame->pose = rawPose;
        return true;
    }

    clock_gettime(CLOCK_REALTIME, &start_dege_prepare);
    trackingSingleFrame(pFrame, bestPose, 0, 0);
    clock_gettime(CLOCK_REALTIME, &end_dege_prepare);
    prepare_dege_corr = ((double)end_dege_prepare.tv_nsec - (double)start_dege_prepare.tv_nsec) / 1000000.0;
    cout << "-----------------prepare time2 :" << prepare_dege_corr << " ms" << endl;

    cout << "correct pose: " << rawPose.trans(0) << ", " << rawPose.trans(1) << ", " << rawPose.theta;
    cout << " ---> " << pFrame->pose.trans(0) << ", " << pFrame->pose.trans(1) << ", " << pFrame->pose.theta << endl;
    cout << "[DegeCorrect] correct successfully!" << endl;

    return true;
}
// **extract the line near the reftheta**
// **record the pixel which is near the reftheta into the pFrameCorrect_xy**
// gaussianblur the mask ,detect edge using sobel to get the Mat: dx dy
// traverse the pixel in mask, record the pixel which theta is close to the reftheta
void Tracking::getFrameCorrectBuff(cv::Mat &Mask, float refTheta)
{

    cv::Mat mask = Mask * 255.0f;
    mask.convertTo(mask, CV_8UC1);

    cv::Mat gaussMask;
    cv::GaussianBlur(mask, gaussMask, cv::Size(3, 3), 3, 3);

    cv::Mat dx, dy;
    cv::Sobel(gaussMask, dx, CV_16S, 0, 1); //???? why don't use gaussMaskList
    cv::Sobel(gaussMask, dy, CV_16S, 1, 0);

    int count = 0;
    for (int i = 0; i < mask.rows; i++)
    {

        unsigned char *maskPtr = mask.ptr<unsigned char>(i);
        short *dxPtr = dx.ptr<short>(i);
        short *dyPtr = dy.ptr<short>(i);

        for (int j = 0; j < mask.cols; j++)
        {

            if (maskPtr[j] == 0)
                continue;

            float floatDx = (float)dxPtr[j];
            float floatDy = (float)dyPtr[j];

            // compute the point gradient theta
            // if the point gradient theta is similar the frame dege theta
            // record the point
            float theta = atan(floatDy / floatDx);

            if (abs(getDeltaTheta(theta, refTheta)) < 0.7f)
            {

                pFrameCorrect_xy[count * 2] = i + 0.5f;
                pFrameCorrect_xy[count * 2 + 1] = j + 0.5f;

                count++;
            }
            else
            {
                maskPtr[j] = 0;
            }
        }
    }

    corPixelCount = count;
    // cv::imwrite("selected.png", mask);
}
// create new pose acroding the deltaSpan,
// for each pose ,calc the error between pose and mapList[searchLevel]
// mapList[searchLevel] is a pic ,only have pixel which dirtect near the refTheta
void Tracking::getCandidatePose(Frame *pFrame, float degeSpan, float degeTheta)
{

    candidatePoses.clear();
    candidateSpans.clear();

    float correctStep = 0.5f * pMap->gaussSize[searchLevel];
    float factor = pow(2, -searchLevel);

    bool minSearch = false;
    float minErr = corPixelCount;
    Pose bestPose;
    float bestSpan;

    for (float deltaTrans = -degeSpan; deltaTrans < degeSpan; deltaTrans += correctStep)
    {

        float trans_x = pFrame->pose.trans(0) - deltaTrans * cos(degeTheta);
        float trans_y = pFrame->pose.trans(1) - deltaTrans * sin(degeTheta);

        Pose pose(trans_x, trans_y, pFrame->pose.theta);
        // Pose pose(trans_x * factor, trans_y * factor, pFrame->pose.theta);

        float error = calcCorrectError(pose, pMap->mapList[searchLevel]); //???? error is number, the calcCorrectError small the errro small too.

        cout << "correct error count: " << error << " / " << corPixelCount << "~~~span~~~" << deltaTrans << endl; // corPixelCount is number of pixel which gradtheta near reftheta
        // cout << "dege theta: " << degeTheta << endl;
        //find the local minimum
        if (error < minErr)
        {
            if (error * 4.0f < (float)corPixelCount)
            { //????? error*4????
                minSearch = true;
                minErr = error;
                bestPose = Pose(trans_x, trans_y, pFrame->pose.theta);
                bestSpan = deltaTrans;
            }
        }
        else
        {

            if (minSearch)
            {

                //cout<<"push back pose..."<<endl;
                candidatePoses.push_back(bestPose);
                candidateSpans.push_back(bestSpan / degeSpan);

                minSearch = false;
                minErr = corPixelCount;
            }
        }
    }

    cout << "candidate pose num: " << candidatePoses.size() << endl;
}
// using pose to get the pixel in map and  pFrameCorrect_xy(the pixel near the reftheta)
// if the pixel is small enough, resduals+=1
float Tracking::calcCorrectError(Pose &pose, cv::Mat &map)
{
    float factor = pow(2, -searchLevel);

    float tx = pose.trans(0) * factor;
    float ty = pose.trans(1) * factor;

    float cosTheta = cos(pose.theta);
    float sinTheta = sin(pose.theta);

    float residuals = 0.0f;
    for (int i = 0; i < corPixelCount; i++)
    {

        float x = pFrameCorrect_xy[2 * i];
        float y = pFrameCorrect_xy[2 * i + 1];

        int map_x = (int)(cosTheta * x - sinTheta * y + tx + 0.5f);
        int map_y = (int)(sinTheta * x + cosTheta * y + ty + 0.5f);

        if (map_x < 0 || map_y < 0 || map_x >= map.rows || map_y >= map.cols)
        {
            continue;
        }

        float mapPixel = map.ptr<float>(map_x)[map_y];

        if (mapPixel > 0.1f) //??????
            continue;

        residuals += 1.0f;
    }

    return residuals;
}

float Tracking::calcBidirectError(Frame *pFrame, int level)
{

    cv::Mat &map = pMap->mapList[level];
    cv::Mat &frame = pFrame->gaussMaskList[level];

    float factor = pow(2, -level);
    float tx = pFrame->pose.trans(0) * factor;
    float ty = pFrame->pose.trans(1) * factor;

    float cosTheta = cos(pFrame->pose.theta);
    float sinTheta = sin(pFrame->pose.theta);

    float residuals = 0.0f;
    for (int i = 0; i < frame.rows; i++)
    {

        float *framePtr = frame.ptr<float>(i);
        for (int j = 0; j < frame.cols; j++)
        {

            float framePixel = framePtr[j];
            if(framePixel < 0.1f) continue;

            float x = i + 0.5f;
            float y = j + 0.5f;

            int map_x = (int)(cosTheta * x - sinTheta * y + tx + 0.5f);
            int map_y = (int)(sinTheta * x + cosTheta * y + ty + 0.5f);

            float mapPixel;
            if (map_x < 0 || map_y < 0 || map_x >= map.rows || map_y >= map.cols)
            {
                mapPixel = 0.0f;
            }
            else
            {
                mapPixel = map.ptr<float>(map_x)[map_y];
            }

            float err = mapPixel - framePixel;
            residuals += err * err;
        }
    }

    return residuals;
}

// get delta (theta1) and set to(-pi/2, pi/2)
float Tracking::getDeltaTheta(float theta1, float theta2)
{

    float delta = theta1 - theta2;

    delta = (delta > 1.5708f) ? (delta - 3.1416f) : delta;
    delta = (delta < -1.5708f) ? (delta + 3.1416f) : delta;

    return delta;
}