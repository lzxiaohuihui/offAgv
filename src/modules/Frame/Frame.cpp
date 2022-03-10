#include "Frame.h"

#include "../Base/ImgProcess.h"

using namespace std;

bool Frame::init(int row, int col, int pyramid_level_){

    if(pyramid_level_ > MAX_PYRAMID_LEVEL){

        cout<<"[Frame::init] error: pyramid level exceed limit"<<endl;
        return false;
    }

    pyramid_level = pyramid_level_;

    maskList.resize(pyramid_level);
    gaussMaskList.resize(pyramid_level);
    gradMaskList.resize(pyramid_level);

    goodCorrectCount = 0;

    rows[0] = row;
    cols[0] = col;
    //scale[0] = scale_;

    for(int l = 1; l < pyramid_level; l++){

        if(rows[l-1]%2 != 0 || rows[l-1]%2 != 0){

            cout<<"[Frame::init] 2^pyramid_level width/height expected"<<endl;
            return false;
        }

        rows[l] = (rows[l-1])/2;
        cols[l] = (cols[l-1])/2;

        //scale[l] = scale[l-1] * 0.5f;
    }

    for(int l = 0; l < pyramid_level; l++){

        maskList[l].create(rows[l], cols[l], CV_32F);

        gaussMaskList[l].create(rows[l], cols[l], CV_32F);
        gradMaskList[l].create(rows[l], cols[l], CV_32FC2);
    }

    cout<<"[Frame::init] init successful"<<endl;

    return true;
}
//init the pyramid(maskList gaussMaskList gradMaskList)
//check the degenerate
void Frame::buildFrame(string imaName){

    // cout<<"[Frame::buildFrame] build current frame..."<<endl;

    lost = false;
    degenerative = false;

    cv::Mat inImage = cv::imread(imaName);
    if(inImage.empty()){
		cout<<"[Frame::buildFrame] error: image'"<<imaName<<"' is not found"<<endl;
		exit(-1);
	}

    if(inImage.channels() == 3)
        cv::cvtColor(inImage, inImage, CV_RGB2GRAY);

    //binarize
    cv::threshold(inImage, rawMask, 50, 255, CV_THRESH_BINARY);

    const uchar *rawPtr = rawMask.data;
    float *listPtr = (float*)maskList[0].data;

    int pixelNum = rawMask.rows * rawMask.cols;
    for(int i = 0; i < pixelNum; i++){

        listPtr[0] = (rawPtr[0] > 30) ? 1.0f : 0.0f;

        listPtr++;
        rawPtr++ ;
    }

    for(int l = 1; l < pyramid_level; l++)
        ImgProcess::imageResizeHalf(maskList[l-1], maskList[l]);
    
    
    for(int l = 0; l < pyramid_level; l++){

        cv::GaussianBlur(maskList[l], gaussMaskList[l], 
                        cv::Size(gaussSize[l], gaussSize[l]), gaussSigma[l], gaussSigma[l]);

        // cv::Mat img = gaussMaskList[l] * 255.0f;
        // img.convertTo(img, CV_8UC1);
        // string imaName = "gaussFrame_" + to_string(l) + ".png";
        // cv::imwrite(imaName, img);

        ImgProcess::computeGradientDiff(gaussMaskList[l], gradMaskList[l]);

    }

    if(!linesFit()){
        cout<<"[Frame::buildFrame] lose..."<<endl;
        lost = true;
        return;
    }

    if(degenerateCheck()){
        cout<<"[Frame::buildFrame] degenerate..."<<endl;
        degenerative = true;
    }

    return;
}


bool Frame::linesFit(){

    for(int i = 0; i < laneLines.size(); i++)
        delete laneLines[i];
    laneLines.clear();

    cv::Mat img = maskList[pyramid_level-1] * 255.0f;
    img.convertTo(img, CV_8UC1);
    cv::GaussianBlur(img, img, cv::Size(5, 5), 5, 5);

    cv::threshold(img, img, 80, 255, CV_THRESH_BINARY);

    cv::Mat edgeImage;
    cv::Canny(img, edgeImage, 150, 100, 3);

    vector<cv::Vec4i> lines;
	HoughLinesP(edgeImage, lines, 1, CV_PI/180, 10, 10, 5);

    // cv::cvtColor(edgeImage, edgeImage, cv::COLOR_GRAY2RGB);

    if(lines.size() == 0){
        cout<<"[Frame] can not detect any lines"<<endl;
        return false;
    }
    //cout<<"lines num: "<<lines.size()<<endl;

    // cv::imshow("hough", edgeImage);
    // cv::waitKey(1);

	for( size_t i = 0; i < lines.size(); i++ ){

        cv::Vec4i& l = lines[i];

		Eigen::Vector3f pt1(l[0], l[1], 1.0f);
        Eigen::Vector3f pt2(l[2], l[3], 1.0f);

        float length = (pt1 - pt2).norm();

        Eigen::Vector3f line = pt1.cross(pt2);

        float inv_norm = 1.0f / sqrt(line(0)*line(0) + line(1)*line(1));
        line = inv_norm*line;

        LaneLine* pLine = new LaneLine(line, length);
        laneLines.push_back(pLine);

		//cv::line( edgeImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, CV_AA);
	}

    // cv::Mat pic;
    // cv::resize(edgeImage,pic,cv::Size(640,480),cv::INTER_LINEAR);
    // cv::line( pic, cv::Point(0, 0), cv::Point(300, 300), cv::Scalar(0,0,255), 3, CV_AA);
    // cv::imshow("line", pic);
    // cv::waitKey(1);


    // cv::Mat pic;
    // cv::resize(edgeImage,pic,cv::Size(64,48),cv::INTER_LINEAR);
    // cv::imshow("sd",edgeImage);


    return true;
}

bool Frame::degenerateCheck(){

    assert(laneLines.size() > 0);

    if(laneLines.size() == 1)
        return true;

    float refTheta = laneLines[0]->theta;

    float maxDelta = 0.0f;
    float minDelta = 0.0f;
    vector<float> deltaThetas;

    for(vector<LaneLine*>::iterator iter = laneLines.begin(); iter < laneLines.end(); iter++){

        float curTheta = (*iter)->theta;
        float delta = getDeltaTheta(curTheta, refTheta);

        deltaThetas.push_back(delta);

        maxDelta = (maxDelta > delta) ? maxDelta : delta;
        minDelta = (minDelta < delta) ? minDelta : delta;

        if(maxDelta-minDelta > 0.3f)
            return false;
    }

    assert(laneLines.size() == deltaThetas.size());

    float totalTheta = 0.0f;
    float totalWeight = 0.0f;
    for(int i = 0; i < laneLines.size(); i++){

        float weight = laneLines[i]->length / 10.0f;

        totalTheta += weight * (refTheta + deltaThetas[i]);
        totalWeight += weight;
    }

    degeTheta = totalTheta / totalWeight;
    
    return true;

}
// check whether the frame is a undege pic
bool Frame::correctCheck(float degeTheta){

    float refTheta = degeTheta - pose.theta;

    int goodCount = 0;
    for(vector<LaneLine*>::iterator iter = laneLines.begin(); iter < laneLines.end(); iter++){

        float curTheta = (*iter)->theta;
        float delta = getDeltaTheta(curTheta, refTheta);

        if(abs(delta) > 0.5f && (*iter)->length > 10)
            goodCount++;

        if(goodCount > 1)
            break;
    }

    if(goodCount > 1){

        goodCorrectCount++;

        if(goodCorrectCount > 5){
            goodCorrectCount = 0;
            return true;
        }

    }else{

        goodCorrectCount = 0;
    }

    return false;
}

float Frame::getDeltaTheta(float theta1, float theta2){

    float delta = theta1 - theta2;

    delta = (delta > 1.5708f) ? (delta - 3.1416f) : delta;
    delta = (delta < -1.5708f) ? (delta + 3.1416f) : delta;

    return delta;

}



