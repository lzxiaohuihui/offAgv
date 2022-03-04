
#include "Tracking.h"

#include "../Base/ImgProcess.h"

using namespace std;
using namespace cv;

Tracking::Tracking(Map *pMap_) : pMap(pMap_),
                                 inited(false),
                                 noEmptyPixels(0),
                                 validPixels(0),
                                 maxIterNum(5),
                                 smoothWeight(1.0f),
                                 thetaWeight(100.0f),
                                 firstOpt(true)
{

    pFrame_xyi = nullptr;
    pMap_xy = nullptr;
    pMapGrad_xy = nullptr;
    pRp_xy = nullptr;
    pRes = nullptr;
}

Tracking::~Tracking()
{

    if (inited)
    {

        delete pFrame_xyi;
        delete pMap_xy;
        delete pMapGrad_xy;
        delete pRp_xy;
        delete pRes;

        delete pFrameCorrect_xy;
    }
}

bool Tracking::init(int frameRows_, int frameCols_, int pyramid_level_)
{

    if (inited)
    {
        cout << "[Tracking::init] already inited " << endl;
        return false;
    }

    if (pyramid_level_ <= 0 || pyramid_level_ > MAX_PYRAMID_LEVEL)
    {

        cout << "[Tracking::init] error: invalid param pyramid_level " << pyramid_level_ << endl;
        return false;
    }

    pyramid_level = pyramid_level_;

    if (frameRows_ <= 0 || frameCols_ <= 0)
    {
        cout << "[Tracking::init] error: invalid frame size: " << frameRows_ << ", " << frameCols_ << endl;
        return false;
    }

    frameRows = frameRows_;
    frameCols = frameCols_;

    OPTIMIZE_BUFFER_SIZE = frameRows * frameCols;

    pFrame_xyi = new float[OPTIMIZE_BUFFER_SIZE * 3];
    pMap_xy = new float[OPTIMIZE_BUFFER_SIZE * 2];
    pMapGrad_xy = new float[OPTIMIZE_BUFFER_SIZE * 2];
    pRp_xy = new float[OPTIMIZE_BUFFER_SIZE * 2];
    pRes = new float[OPTIMIZE_BUFFER_SIZE];

    pFrameCorrect_xy = new float[OPTIMIZE_BUFFER_SIZE * 2];

    searchLevel = pyramid_level - 1;
    optimateLevel = pyramid_level - 1;
    evaluLevel = 0;

    cout << "[Tracking::init] init successful" << endl;

    inited = true;
    return true;
}

bool Tracking::getLocalMap(const cv::Mat &temp_frame, Eigen::Vector3f params, vector<cv::Point2i> &local_map, vector<cv::Point2i> &icp_frame)
{
    // 提取局部地图

    cv::Mat temp_map = pMap->mapList[0];
    // cv::Mat edge_map(temp_map.rows, temp_map.cols, 0);
    // for (int i = 0; i < temp_map.rows; ++i){
    //     for(int j = 0; j < temp_map.cols; ++j){
    //         edge_map.at<uchar>(i, j) = temp_map.at<float>(i, j);
    //     }
    // }
    temp_map.convertTo(temp_map, CV_8UC1);
    int edge_thresh = 50;
    cv::Canny(temp_frame, temp_frame, edge_thresh, edge_thresh * 3);
    // cv::Canny(temp_map, temp_map, edge_thresh, edge_thresh * 3);

    float temp_tx = params(0);
    float temp_ty = params(1);
    float temp_sinTheta = sin(params(2));
    float temp_cosTheta = cos(params(2));
    // cout << "get params." << endl;
    // cout << params(0) << endl;
    // cout << params(1) << endl;
    // cout << params(2) << endl;

    int count = 0;
    for (int i = 0; i < temp_frame.rows; ++i)
    {
        for (int j = 0; j < temp_frame.cols; ++j)
        {
            float Rp_x = temp_cosTheta * i - temp_sinTheta * j;
            float Rp_y = temp_sinTheta * i + temp_cosTheta * j;

            // 地图点 Rx + t
            int map_x = (int)(Rp_x + temp_tx);
            int map_y = (int)(Rp_y + temp_ty);

            // 判断地图边界
            if (map_x < 0 || map_y < 0 || map_x >= temp_map.rows || map_y >= temp_map.cols)
                continue;

            if (temp_frame.at<uchar>(i, j) != 0)
            {
                icp_frame.push_back(cv::Point(map_x, map_y));
            }

            if (temp_map.at<uchar>(map_x, map_y) != 0)
            {
                local_map.push_back(cv::Point(map_x, map_y));
                count++;
            }
            // cout << temp_pixel << endl;
        }
    }
    // cout << "temp map size: " << temp_map.size() << endl;
    cout << "temp frame type is: " << temp_frame.type() << "---" << temp_map.type() << endl;
    cout << "++++++++++++++icp_map and local_map size:" << icp_frame.size() << "\t" << local_map.size() << endl;

    // show local map
    // cv::Mat show_local_map;
    // show_local_map.create(temp_map.rows, temp_map.cols, CV_8UC1);
    // cout << show_local_map.type() << endl;
    // for (int i = 0; i < local_map.size(); ++i)
    // {
    //     show_local_map.at<uchar>(local_map.at(i).x, local_map.at(i).y) = 255;
    // }

    // cout << "local_map_x_y:" << show_local_map.size << endl;

    cv::namedWindow("map", 0);
    cv::resizeWindow("map", 800, 800);
    cv::imshow("map", temp_map);
    cv::waitKey(10);

    return count > MIN_NOEMPTY_PIXEL;
}

bool Tracking::icpTracking(Frame *pFrame, Pose &PriPose)
{
    assert(pFrame != nullptr);

    curFrame = pFrame;
    curPose = PriPose;
    constPose = PriPose;

    // float resErr;

    // firstOpt = true;

    pose2Params(0);

    // getNoEmptyPixels(pFrame->gaussMaskList[0]);

    // if (noEmptyPixels < (MIN_NOEMPTY_PIXEL))
    // {

    //     cout << "[Tracking::trackingSingleFrame] non-empty pixel count too small: " << noEmptyPixels << endl;
    //     pFrame->pose = curPose;
    //     return false;
    // }

    vector<cv::Point> local_map;
    vector<cv::Point> icp_frame;
    if (!getLocalMap(pFrame->rawMask, curParams, local_map, icp_frame))
    {
        pFrame->pose = curPose;
        cout << "[Tracking::trackingSingleFrame] non-empty pixel count too small: " << endl;
        return false;
    }
    // cout << "icp_map and local_map size:" << icp_frame.size()<<"/"<<local_map.size() << endl;

    // icpTrackingWithKdtree(local_map, icp_frame);
    icpTrackingWithG_N(local_map, icp_frame);

    params2Pose(0);

    pFrame->pose = curPose;
    return true;
}

// according to the PriPose ,get a pic of frame ---- curFrame
// using the error between the curFrame and framePixel to optimate pose
// 1. traverse the level of pyramid
// 2. collect the pixel which is no empty
// 3. calc the residual error on these no empty pixel
// 4. LM optimate the pose，update the pose
bool Tracking::trackingSingleFrame(Frame *pFrame, Pose &PriPose, int startLevel, int endLevel)
{

    cout << "Tracking.." << endl;
    assert(pFrame != nullptr);

    assert(startLevel >= 0 && startLevel < pyramid_level &&
           endLevel >= 0 && endLevel < pyramid_level);

    assert(startLevel >= endLevel);

    smoothWeight = (pFrame->degenerative) ? 10.0f : 1.0f;

    curPose = PriPose;
    constPose = PriPose;

    float resErr;

    for (int level = startLevel; level >= endLevel; level--)
    {

        firstOpt = true;

        pose2Params(level);

        getNoEmptyPixels(pFrame->gaussMaskList[level]);

        if (noEmptyPixels < (MIN_NOEMPTY_PIXEL >> (level * 2)))
        {

            cout << "[Tracking::trackingSingleFrame] non-empty pixel count too small: " << noEmptyPixels << endl;
            pFrame->pose = curPose;
            return false;
        }

        resErr = calcResidualError(curParams, level); //0.02s

        LM_Optimate(level, resErr);

        params2Pose(level);
    }

    pFrame->pose = curPose;

    if (false)
    {

        cv::Mat demo = ImgProcess::drawFrameToMap(pFrame->gaussMaskList[0], pMap->mapList[0], pFrame->pose);
        cv::imshow("optimized frame", demo);
        cv::waitKey(10);
        //getchar();
    }

    return true;
}

void Tracking::pose2Params(int level)
{

    assert(level >= 0);

    float factor = pow(2, -level);

    Eigen::Vector2f params_trans = curPose.trans * factor;

    curParams(0) = params_trans(0);
    curParams(1) = params_trans(1);

    curParams(2) = curPose.theta;
}

void Tracking::params2Pose(int level)
{

    assert(level >= 0);

    float factor = pow(2, level);

    Eigen::Vector2f params_trans(curParams(0), curParams(1));
    curPose.trans = params_trans * factor;

    curPose.theta = curParams(2);
}

void Tracking::LM_Optimate(int level, float &refErr)
{

    float lamda = 0.0f;
    int iterCount = 0;

    while (iterCount < maxIterNum)
    {

        buildNormEquation(level); // 0.005s

        int tryCount = 0;
        while (true)
        {
            //search lamda

            Eigen::Matrix3f A = JtJ;
            Eigen::Vector3f b = Jtr;

            // cout<<"A: "<<A<<endl;
            // cout<<"b: "<<b<<endl;

            for (int i = 0; i < 3; i++)
                A(i, i) += lamda;

            Eigen::Vector3f deltaParams = A.ldlt().solve(b);

            Eigen::Vector3f newParams = curParams + deltaParams;

            float newErr = calcResidualError(newParams, level);

            if (newErr < refErr)
            {

                //cout<<"accept lamda..."<<endl;

                //accept lamda
                if (lamda < 0.1f)
                    lamda = 0.0f;
                else
                    lamda *= (1.0f / 8.0f);

                if (newErr / refErr > 0.9995)
                    iterCount = maxIterNum;

                curParams = newParams;
                refErr = newErr;

                break;
            }
            else
            {

                //cout<<"reject lamda..."<<endl;

                //reject lamda
                if (lamda < 0.1f)
                    lamda = 0.1f;
                else
                    lamda *= 8.0f;

                if (deltaParams.norm() < 0.000001f)
                {
                    iterCount = maxIterNum;
                    break;
                }

                if (tryCount > 30)
                {
                    iterCount = maxIterNum;
                    break;
                }
            }
            tryCount++;
        }

        iterCount++;
    }
}

void Tracking::getNoEmptyPixels(const cv::Mat &frame)
{

    int count = 0;

    int rowStep = 1; //2
    int colStep = 1;
    for (int i = 0; i < frame.rows; i += rowStep)
    {

        const float *rowPtr = frame.ptr<float>(i);
        for (int j = 0; j < frame.cols; j += colStep)
        {

            float pixel = rowPtr[j];

            if (pixel < 0.1f)
                continue;

            pFrame_xyi[count * 3] = i + 0.5f;
            pFrame_xyi[count * 3 + 1] = j + 0.5f;
            pFrame_xyi[count * 3 + 2] = pixel;

            count++;
        }
    }
    noEmptyPixels = count;

    // cout<<"non-empty pixels: "<<count<<endl;
    // getchar();
}

float Tracking::calcResidualError(Eigen::Vector3f params, int level)
{

    cv::Mat &map = pMap->mapList[level];
    cv::Mat &mapGrad = pMap->gradList[level];

    // Eigen::Vector2f trans(params(0), params(1));

    // float theta = params(2);
    // Eigen::Matrix2f Rota;
    // Rota<<cos(theta), -sin(theta), sin(theta), cos(theta);

    float tx = params(0);
    float ty = params(1);

    //cout<<"***  "<<tx<<"  "<<ty<<"  "<<params(2)<<endl;
    float cosTheta = cos(params(2));
    float sinTheta = sin(params(2));

    float residuals = 0.0f;
    int validCount = 0;
    for (int i = 0; i < noEmptyPixels; i++)
    {

        float x = pFrame_xyi[i * 3];
        float y = pFrame_xyi[i * 3 + 1];
        float framePixel = pFrame_xyi[i * 3 + 2];

        // Eigen::Vector2f pt(x, y);
        // Eigen::Vector2f Rp = Rota * pt;
        // pt = Rp + trans;

        // int map_x = (int)(pt(0) + 0.5f);
        // int map_y = (int)(pt(1) + 0.5f);

        float Rp_x = cosTheta * x - sinTheta * y;
        float Rp_y = sinTheta * x + cosTheta * y;

        int map_x = (int)(Rp_x + tx + 0.5f);
        int map_y = (int)(Rp_y + ty + 0.5f);

        if (map_x < 0 || map_y < 0 || map_x >= map.rows || map_y >= map.cols)
        {

            pRes[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        float res = map.ptr<float>(map_x)[map_y] - framePixel;
        pRes[i] = res;
        residuals += res * res;

        Float2 grad = mapGrad.ptr<Float2>(map_x)[map_y];

        pMapGrad_xy[i * 2] = grad.x;
        pMapGrad_xy[i * 2 + 1] = grad.y;

        // pRp_xy[i*2] = Rp(0);
        // pRp_xy[i*2+1] = Rp(1);

        if (firstOpt)
        {
            pRp_xy[i * 2] = Rp_x;
            pRp_xy[i * 2 + 1] = Rp_y;
        }

        validCount++;
    }

    // use FEJ?
    // if(firstOpt)
    //     firstOpt = false;

    float factor = pow(4, -level);

    smoResVec(0) = constPose.trans(0) - params(0);
    smoResVec(1) = constPose.trans(1) - params(1);
    smoResVec(2) = thetaWeight * (constPose.theta - params(2));

    smoResVec = factor * smoothWeight * smoResVec;

    residuals += smoResVec.norm();

    validPixels = validCount;

    return residuals;
}

void Tracking::buildNormEquation(int level)
{

    memset(JtJ.data(), 0, sizeof(float) * 9);
    memset(Jtr.data(), 0, sizeof(float) * 3);

    float factor = pow(4, -level);

    //data
    for (int i = 0; i < noEmptyPixels; i++)
    {

        float res = pRes[i];

        if (isnanf(res))
            continue;

        float grad_x = pMapGrad_xy[i * 2];
        float grad_y = pMapGrad_xy[i * 2 + 1];

        float Rp_x = pRp_xy[i * 2];
        float Rp_y = pRp_xy[i * 2 + 1];

        float j_tx = grad_x;
        float j_ty = grad_y;
        float j_theta = (-grad_x * Rp_y + grad_y * Rp_x);

        JtJ(0, 0) += j_tx * j_tx;
        JtJ(0, 1) += j_tx * j_ty;
        JtJ(0, 2) += j_tx * j_theta;
        JtJ(1, 1) += j_ty * j_ty;
        JtJ(1, 2) += j_ty * j_theta;
        JtJ(2, 2) += j_theta * j_theta;

        Jtr(0) -= j_tx * res;
        Jtr(1) -= j_ty * res;
        Jtr(2) -= j_theta * res;
    }

    //smooth
    float j_tx = factor * smoothWeight;
    float j_ty = factor * smoothWeight;
    float j_theta = factor * smoothWeight * thetaWeight;

    JtJ(0, 0) += j_tx * j_tx;
    JtJ(0, 1) += j_tx * j_ty;
    JtJ(0, 2) += j_tx * j_theta;
    JtJ(1, 1) += j_ty * j_ty;
    JtJ(1, 2) += j_ty * j_theta;
    JtJ(2, 2) += j_theta * j_theta;

    Jtr(0) -= j_tx * smoResVec(0);
    Jtr(1) -= j_ty * smoResVec(1);
    Jtr(2) -= j_theta * smoResVec(2);

    JtJ(1, 0) = JtJ(0, 1);
    JtJ(2, 0) = JtJ(0, 2);
    JtJ(2, 1) = JtJ(1, 2);
}

bool Tracking::downSample(vector<cv::Point> &ref_pt_vec, vector<cv::Point> &new_pt_vec, cv::Mat &ICP_ref_pts, cv::Mat &ICP_new_pts)
{
    int ref_pt_num, new_pt_num;
    ref_pt_num = ref_pt_vec.size();
    new_pt_num = new_pt_vec.size();

    ICP_ref_pts = cv::Mat(ref_pt_num / REF_DOWN_SAMPLE_RATE, 2, CV_32FC1);
    ICP_new_pts = cv::Mat(new_pt_num / NEW_DOWN_SAMPLE_RATE, 2, CV_32FC1);

    for (int i = 0; i < ref_pt_num / REF_DOWN_SAMPLE_RATE; i++)
    {
        ICP_ref_pts.at<float>(i, 0) = ref_pt_vec[i * REF_DOWN_SAMPLE_RATE].x;
        ICP_ref_pts.at<float>(i, 1) = ref_pt_vec[i * REF_DOWN_SAMPLE_RATE].y;
    }
    for (int i = 0; i < new_pt_num / NEW_DOWN_SAMPLE_RATE; i++)
    {
        ICP_new_pts.at<float>(i, 0) = new_pt_vec[i * NEW_DOWN_SAMPLE_RATE].x;
        ICP_new_pts.at<float>(i, 1) = new_pt_vec[i * NEW_DOWN_SAMPLE_RATE].y;
    }

    return true;
}
bool Tracking::findDataAssociation(cv::Mat &ICP_ref_pts, cv::Mat &ICP_new_pts, vector<cv::Point> &exist_association_pts, vector<cv::Point> &nearest_pts, int knn)
{
    vector<int> near_point_index;
    vector<float> near_point_distance;

    float cos_theta = cos(constPose.theta);
    float sin_theta = sin(constPose.theta);

    for (int i = 0; i < ICP_new_pts.rows; i++)
    {
        My_Kdtree.knnSearch(ICP_new_pts.row(i), near_point_index, near_point_distance, knn, cv::flann::SearchParams(-1));
        float near_x = 0.0f, near_y = 0.0f;
        for (int index : near_point_index)
        {
            near_x = ICP_ref_pts.at<float>(index, 0);
            near_y = ICP_ref_pts.at<float>(index, 1);
        }
        // compute normal difference
        float mapNormalx = pMap->icpGradMap.at<Float2>(near_x, near_y).x;
        float mapNormaly = pMap->icpGradMap.at<Float2>(near_x, near_y).y;

        // transform map_point to frame
        float frame_x = cos_theta * (ICP_new_pts.at<float>(i, 0) - constPose.trans(0)) + sin_theta * (ICP_new_pts.at<float>(i, 1) - constPose.trans(1));
        float frame_y = -sin_theta * (ICP_new_pts.at<float>(i, 0) - constPose.trans(0)) + cos_theta * (ICP_new_pts.at<float>(i, 1) - constPose.trans(1));

        // get frame normal
        float frameNormalx = curFrame->gradMaskList[0].at<Float2>(frame_x, frame_y).x;
        float frameNormaly = curFrame->gradMaskList[0].at<Float2>(frame_x, frame_y).y;

        // rotate frame normal to map
        float frameToMapNormalx = cos_theta * frameNormalx - sin_theta * frameNormaly;
        float frameToMapNormaly = sin_theta * frameNormalx + cos_theta * frameNormaly;

        // compute the difference of two normals

        float err1 = pow(mapNormalx - frameToMapNormalx, 2);
        float err2 = pow(mapNormaly - frameToMapNormaly, 2);

        float err = sqrt(err1 + err2);
        // cout << err << endl;
        if (err > NORMAL_THRESH)
            continue;

        exist_association_pts.push_back(cv::Point(ICP_new_pts.at<float>(i, 0), ICP_new_pts.at<float>(i, 1)));
        nearest_pts.push_back(cv::Point(near_x, near_y));
    }

    return true;
}

bool Tracking::icpTrackingWithG_N(vector<cv::Point> &ref_pt_vec, vector<cv::Point> &new_pt_vec)
{
    float total_tx = 0.0f, total_ty = 0.0f, total_theta = 0.0f;
    // float res_tx = 0.0f, res_ty = 0.0f;
    // float res_theta = 0.0f;
    // float cos_res_theta = cos(res_theta);
    // float sin_res_theta = sin(res_theta);

    cv::Mat ICP_ref_pts, ICP_new_pts;
    downSample(ref_pt_vec, new_pt_vec, ICP_ref_pts, ICP_new_pts);
    // cout << "++++++++++++++++++frame non-pixels num: " << ICP_new_pts.rows << endl;
    My_Kdtree.build(ICP_ref_pts, flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);

    float iter_err, last_iter_err;
    for (int iter_num = 0; iter_num < ITER_MAX; ++iter_num)
    {
        vector<cv::Point> exist_association_pts;
        vector<cv::Point> nearest_pts;

        struct timeval time;
        gettimeofday(&time, NULL);
        double start = ((double)time.tv_sec + time.tv_usec / 1000.0);

        findDataAssociation(ICP_ref_pts, ICP_new_pts, exist_association_pts, nearest_pts, 1);

        gettimeofday(&time, NULL);
        double end = ((double)time.tv_sec + time.tv_usec / 1000.0);

        cout << "--------------------dataAssociation number: " << exist_association_pts.size() << endl;
        std::cout << "-----------------findDataAssociation time :" << end - start << " ms" << std::endl;
        if (exist_association_pts.size() < 30)
            return false;

        Eigen::Matrix3d hessian;
        Eigen::Vector3d jaco;
        Eigen::Vector3d bias;
        for (int i = 0; i < exist_association_pts.size(); ++i)
        {
            int u = exist_association_pts[i].x;
            int v = exist_association_pts[i].y;
            int x = nearest_pts[i].x;
            int y = nearest_pts[i].y;
            float alpha = pMap->icpGradMap.at<Float2>(x, y).x;
            float beta = pMap->icpGradMap.at<Float2>(x, y).y;
            float error_func = alpha * (x - u) + beta * (y - v);
            float j3 = alpha * v - beta * u;

            jaco = Eigen::Vector3d(-alpha, -beta, j3);
            hessian += jaco * jaco.transpose();
            bias += -error_func * jaco;
            iter_err += error_func * error_func;
            // cout << "grad.x_y: " << atan(beta / alpha) << endl;
        }

        Eigen::Vector3d update = hessian.ldlt().solve(bias);
        cout << "update. " << endl;
        cout << update[0] << endl;
        cout << update[1] << endl;
        cout << update[2] << endl;

        // update init pose
        // res_theta += update[2];
        // cos_res_theta = cos(res_theta);
        // sin_res_theta = sin(res_theta);
        // res_tx = cos_res_theta * res_tx - sin_res_theta * res_ty + update[0];
        // res_ty = sin_res_theta * res_tx + sin_res_theta * res_ty + update[1];

        float sin_delta = sin(update[2]);
        float cos_delta = cos(update[2]);
        total_theta += update[2];
        total_tx = cos_delta * total_tx - sin_delta * total_ty + update[0];
        total_ty = sin_delta * total_tx + cos_delta * total_ty + update[1];

        // update icp data
        for (int i = 0; i < ICP_new_pts.rows; i++)
        {
            float x = ICP_new_pts.at<float>(i, 0);
            float y = ICP_new_pts.at<float>(i, 1);
            ICP_new_pts.at<float>(i, 0) = cos_delta * x - sin_delta * y + update[0];
            ICP_new_pts.at<float>(i, 1) = sin_delta * x + cos_delta * y + update[1];
        }
        if (std::isnan(update[0]))
        {
            cout << "update is nan" << endl;
            return false;
        }
        if (iter_num > 0 && iter_err > last_iter_err)
        {
            cout << "cost increased: " << iter_err << ", " << last_iter_err << endl;
            // if (update.norm() > 100)
            // {
            //     cout << "update is too big" << endl;
            //     return false;
            // }

            break;
        }
        if (update.norm() < 1e-6)
        {
            cout << "converge. iteration: " << iter_num << endl;
            break;
        }
        cout << "iteration: " << iter_num << ", cost " << iter_err << endl;
        last_iter_err = iter_err;
        iter_err = 0;
    }

    float cos_total_theta = cos(total_theta);
    float sin_total_theta = sin(total_theta);

    curParams(0) = cos_total_theta * curParams(0) - sin_total_theta * curParams(1) + total_tx;
    curParams(1) = sin_total_theta * curParams(0) + cos_total_theta * curParams(1) + total_ty;
    curParams(2) += total_theta;

    // cout << curParams(0) << endl;
    // cout << curParams(1) << endl;
    // cout << curParams(2) << endl;
    return true;
}

bool Tracking::icpTrackingWithKdtree(vector<cv::Point> &ref_pt_vec, vector<cv::Point> &new_pt_vec)
{
    cv::Mat ICP_ref_pts, ICP_new_pts;
    downSample(ref_pt_vec, new_pt_vec, ICP_ref_pts, ICP_new_pts);

    // 1. 建立 k-d tree - 还可以把这个k-d tree保存起来
    // cv::flann::Index My_Kdtree(ICP_ref_pts, flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);
    My_Kdtree.build(ICP_ref_pts, flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);
    // 这里，第一个参数是Mat，第二个参数是建立树的个数，第三个是使用欧式距离
    /**
	 void knnSearch(InputArray query, OutputArray indices,
                   OutputArray dists, int knn, const SearchParams& params=SearchParams());

	query:要查询的矩阵,它的数据类型必须与Input_Mat的类型一致.
    indices:输出的结果的索引
    dists:输出结果对应的距离值,该值为欧式距离的平方.
    knn:索引的top N
    SearchParams:搜索的节点树,如果值为-1,则在树的所有节点进行搜索.
	*/

    //// 2. 在 tree 中查找最近邻点
    //
    //Mat res_ind, res_dist;
    //My_Kdtree.knnSearch(ICP_new_pts.row(0), res_ind, res_dist, 1, flann::SearchParams(-1));

    //cout << "Find nearest index is: " << res_ind << endl;

    //// 让控制台暂停
    //cin.get();

    // 2. ICP迭代
    // 2.1 准备
    float pre_err = FLT_MAX; // 迭代需要的误差量存储1
    float now_err = 0;       // 迭代需要的误差量存储2
    float delta_err;         // 收敛判定依据
    int knn = 1;
    cv::Mat res_ind = cv::Mat(knn, 1, CV_8UC1);
    cv::Mat res_dist = cv::Mat(knn, 1, CV_32FC1); // K-D Tree 查找结果返回值
    int res_ind_int = 0;                          // 查找索引提取
    float near_x, near_y;
    cv::Mat nearest_pts_from_ref = cv::Mat(ICP_new_pts.rows, 2, CV_32FC1);
    cv::Mat mean_new = cv::Mat(1, 2, CV_32FC1);  // ICP 计算过程中求解各个步骤的R、t的中间量，待匹配点云的重心
    cv::Mat mean_near = cv::Mat(1, 2, CV_32FC1); // ICP 计算过程中求解各个步骤的R、t的中间量，迭代中点云的重心
    cv::Mat AXY, BXY;                            // 中间量
    cv::Mat H, U, S, Vt;                         // 中间量
    cv::Mat Mid_eye = cv::Mat::eye(2, 2, CV_32FC1);
    cv::Mat temp_new_pts;
    cv::Mat R, t;         // 阶段计算结果
    cv::Mat R_res, t_res; // 汇总的最终结果

    R_res = cv::Mat::eye(2, 2, CV_32FC1);
    t_res = cv::Mat::zeros(2, 1, CV_32FC1);

    // 为了保留每个new_pts，复制矩阵 ICP_new_pts 至 ICP_new_pts_origin
    cv::Mat ICP_new_pts_origin;
    ICP_new_pts.copyTo(ICP_new_pts_origin);

    // 2.2 迭代开始
    for (int iter_num = 0; iter_num < ITER_MAX; iter_num++)
    {
        now_err = 0;
        for (int i = 0; i < ICP_new_pts.rows; i++)
        {
            // cout << "for i:" << i << endl;
            My_Kdtree.knnSearch(ICP_new_pts.row(i), res_ind, res_dist, knn, cv::flann::SearchParams(-1));
            // cout << "res_ind: " << res_ind.size() << endl;
            // cout << "res_dist: " << res_dist[0] << endl;
            res_ind_int = res_ind.at<int>(0, 0);
            // cout << "near_x_y: " << (res_ind_int < ICP_ref_pts.rows) << endl;
            near_x = ICP_ref_pts.at<float>(res_ind_int, 0);
            near_y = ICP_ref_pts.at<float>(res_ind_int, 1);
            // cout << "near_x_y: " << near_x << "-" << near_y << endl;
            nearest_pts_from_ref.at<float>(i, 0) = near_x;
            nearest_pts_from_ref.at<float>(i, 1) = near_y;

            now_err = now_err + sqrtf((ICP_new_pts.at<float>(i, 0) - near_x) * (ICP_new_pts.at<float>(i, 0) - near_x) +
                                      (ICP_new_pts.at<float>(i, 1) - near_y) * (ICP_new_pts.at<float>(i, 1) - near_y));
        }

        delta_err = pre_err - now_err;
        // cout << "err: " << delta_err << endl;
        if (delta_err < 10)
        {
            break;
        }
        if (delta_err < ITER_THRESH)
        {
            break;
        }
        else
            pre_err = now_err;

        // 求重心，注意：cv::mean 的返回值是一个 cv::scalar 它由四个元素构成，但是我们只用到第一个，所以后面多了个[0]
        mean_new.at<float>(0, 0) = mean(ICP_new_pts.col(0))[0];
        mean_new.at<float>(0, 1) = mean(ICP_new_pts.col(1))[0];
        mean_near.at<float>(0, 0) = mean(nearest_pts_from_ref.col(0))[0];
        mean_near.at<float>(0, 1) = mean(nearest_pts_from_ref.col(1))[0];

        // 所有点按重心归一化
        AXY = ICP_new_pts - repeat(mean_new, ICP_new_pts.rows, 1);
        BXY = nearest_pts_from_ref - repeat(mean_near, nearest_pts_from_ref.rows, 1);

        // 求出待SVD分解的H矩阵
        H = AXY.t() * BXY;
        cv::SVD::compute(H, S, U, Vt);

        // 行列式
        Mid_eye.at<float>(1, 1) = determinant(Vt.t() * U.t());
        R = Vt.t() * Mid_eye * U.t();
        // R = U * Vt;
        t = mean_near.t() - R * mean_new.t();

        // transpose((R * ICP_new_pts.t() + repeat(t, 1, ICP_new_pts.rows)), temp_new_pts);
        // temp_new_pts.copyTo(ICP_new_pts);
        // update icp data
        for (int i = 0; i < ICP_new_pts.rows; i++)
        {
            float x = ICP_new_pts.at<float>(i, 0);
            float y = ICP_new_pts.at<float>(i, 1);
            ICP_new_pts.at<float>(i, 0) = R.at<float>(0, 0) * x + R.at<float>(0, 1) * y + t.at<float>(0, 0);
            ICP_new_pts.at<float>(i, 1) = R.at<float>(1, 0) * x + R.at<float>(1, 1) * y + t.at<float>(1, 0);
        }

        // cout << "temp R-t: " << endl;
        // cout << R << endl;
        // cout << t << endl;
        R_res = R * R_res;
        t_res = R * t_res + t;

        // cout << "1010101010" << endl;
    }
    cout << "err: " << now_err << endl;
    // curParams(0) += t_res.at<float>(0, 0);
    // curParams(1) += t_res.at<float>(1, 0);
    curParams(0) = R_res.at<float>(0, 0) * curParams(0) + R_res.at<float>(0, 1) * curParams(1) + t_res.at<float>(0, 0);
    curParams(1) = R_res.at<float>(1, 0) * curParams(0) + R_res.at<float>(1, 1) * curParams(1) + t_res.at<float>(1, 0);
    curParams(2) += atan(R_res.at<float>(1, 0) / R_res.at<float>(0, 0));

    // cout << curParams(0) << endl;
    // cout << curParams(1) << endl;
    // cout << curParams(2) << endl;

    return true;
}

bool Tracking::csmTracking(Frame *pFrame, Pose &PriPose)
{
    assert(pFrame != nullptr);

    assert(startLevel >= 0 && startLevel < pyramid_level &&
           endLevel >= 0 && endLevel < pyramid_level);

    assert(startLevel >= endLevel);

    smoothWeight = (pFrame->degenerative) ? 10.0f : 1.0f;

    curPose = PriPose;
    constPose = PriPose;

    pose2Params(0);

    getNoEmptyPixels(pFrame->rawMask);

    float bestScore = 0.0f;
    float deltaTx = 100.0f;
    float deltaTy = 100.0f;
    float deltaTheta = 10.0f;

    return true;
}
