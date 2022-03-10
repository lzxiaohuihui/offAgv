
#include "ImgProcess.h"

using namespace std;

bool ImgProcess::computeGradientDiff(const cv::Mat& img, cv::Mat& gradeImg){

    if(img.rows < 2 || img.cols < 2 || img.type() != CV_32FC1)
        return false;
    

    if(gradeImg.rows != img.rows || gradeImg.cols != img.cols ||
            gradeImg.type() != CV_32FC2 || img.type() != CV_32FC1)
        return false;

    // set border
    for(int i = 0; i < img.rows; i += img.rows-1){

        float *rowPtr = (float*)gradeImg.ptr<Float2>(i);
        for(int j = 0; j < img.cols*2; j++)
            rowPtr[i] = 0;

    }
    
    for(int j = 0; j < img.cols; j += img.cols-1){

        for(int i = 0; i < img.rows; i++){

            Float2 *pixel_ptr = ((Float2*)gradeImg.data + (i*img.cols + j));
            pixel_ptr->x = 0.0f;
            pixel_ptr->y = 0.0f;
        }
    }

    // calculate grade
    for(int i = 1; i < img.rows-1; i++){

        float *rowptr = (float*)img.ptr<float>(i);

        float *rowLeftPtr = (float*)img.ptr<float>(i-1);
        float *rowRightPtr = (float*)img.ptr<float>(i+1);

        Float2 *dstPtr = gradeImg.ptr<Float2>(i);
        for(int j = 1; j < img.cols-1; j++){
            
            dstPtr[j].x = 0.5f * (rowRightPtr[j] - rowLeftPtr[j]);
            dstPtr[j].y = 0.5f * (rowptr[j+1] - rowptr[j-1]);
        }

    }
    return true;
}

bool ImgProcess::imageResizeHalf(const cv::Mat& src, cv::Mat& dst){

    if(!(dst.rows*2 == src.rows && dst.cols*2 == src.cols)){
        cout<<"[Map::imageResizeHalf] incorrect image rows/cols"<<endl;
        return false;
    }

    if(!(src.type() == CV_32F && dst.type() == CV_32F)){
        cout<<"[Map::imageResizeHalf] unexpected image tyoe(expected CV_32F)"<<endl;
        return false;
    }

    float *dstPtr = dst.ptr<float>(0);

    for(int i = 0; i < dst.rows; i++){

        const float *srcRow0Ptr = src.ptr<float>(2*i);
        const float *srcRow1Ptr = src.ptr<float>(2*i+1);

        for(int j = 0; j < dst.cols; j++){

            dstPtr[0] = (srcRow0Ptr[0] + srcRow0Ptr[1] + srcRow1Ptr[0] + srcRow1Ptr[1])*0.25f;

            dstPtr ++;
            srcRow0Ptr +=2;
            srcRow1Ptr +=2;
        }

    }

    return true;
}

cv::Mat ImgProcess::drawFrameToMap(const cv::Mat& frame, const cv::Mat& map, Pose& pose){

    cv::Mat demoImg;
    demoImg.create(frame.rows, frame.cols, CV_8UC3);

    float cosTheta = cosf(pose.theta);
    float sinTheta = sinf(pose.theta);

    float tx = pose.trans(0);
    float ty = pose.trans(1);

    // Eigen::Matrix2f Rota;
    // Rota<< cosTheta, -sinTheta, sinTheta, cosTheta;

    for(int i = 0; i < demoImg.rows; i++){

        for(int j = 0; j < demoImg.cols; j++){

            // Eigen::Vector2f pt(i+0.5f, j+0.5f);
            // pt = Rota*pt + pose.trans;

            // int mapRow = (int)(pt(0) + 0.5f);
            // int mapCol = (int)(pt(1) + 0.5f);

            float x = (float)i + 0.5f;
            float y = (float)j + 0.5f;

            float Rp_x = cosTheta*x - sinTheta*y;
            float Rp_y = sinTheta*x + cosTheta*y;

            int mapRow = (int)(Rp_x + tx + 0.5f);
            int mapCol = (int)(Rp_y + ty + 0.5f);

            cv::Vec3b *demoPtr = &demoImg.ptr<cv::Vec3b>(i)[j];

            if(mapRow < 0 || mapCol < 0 || mapRow >= map.rows || mapCol >= map.cols){

                demoPtr->val[0] = 0;
                demoPtr->val[1] = 0;
                demoPtr->val[2] = 0;
                continue;
            }

            float mapPixel = map.ptr<float>(mapRow)[mapCol];
            demoPtr->val[0] = mapPixel * 255;
            demoPtr->val[1] = mapPixel * 255;
            demoPtr->val[2] = mapPixel * 255;

            if(frame.at<float>(i, j) > 0.1){

                demoPtr->val[0] = 0;
                demoPtr->val[1] = 255;
                demoPtr->val[2] = 0;
            }
        }
    }

    return demoImg;
}

bool ImgProcess::poseSwitchPyramid(const int srcLevel, const Pose& srcPose, int dstLevel, Pose& dstPose)
{
	if(srcLevel <0 || dstLevel < 0){

		cout<<"[Map::poseSwitchPyramid] invalid params"<<endl;
		return false;
	}

    float factor = pow(2, srcLevel - dstLevel);

    dstPose.trans = srcPose.trans * factor;
	dstPose.theta = srcPose.theta;

	return true;
}


double ImgProcess::getTime(){

	// struct timeval tv;
	// struct timezone tz;

	// if(gettimeofday(&tv,&tz) ==0)
	// 	return (double)tv.tv_sec*1000.0 + ((double)tv.tv_usec)*1.0e-3;

    
    struct timeval time;

	gettimeofday(&time, NULL);
	return ((double)time.tv_sec + time.tv_usec/1000000.0);

    // return (double)clock()/CLOCKS_PER_SEC;	
}