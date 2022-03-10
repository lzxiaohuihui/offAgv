
#include "Viewer.h"


#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

#include<unistd.h>

using namespace std;


Viewer::Viewer():
                endSwitch(false),
                agvCenter(320.0f, 280.0f),
                T_SE3(Eigen::Matrix4f::Identity()),
                frameRows(580),frameCols(560),
                mapRows(2000),mapCols(2000){

    // demoMap = initMap.clone();
    // if(demoMap.channels() != 3)
    //     cv::cvtColor(demoMap, demoMap, CV_GRAY2RGB);

    overlapMask.create(frameRows, frameCols, CV_8UC1);
    curMask.create(frameRows, frameCols, CV_8UC1);
    
}

void Viewer::updateDemo(Pose& pose, cv::Mat& mask, cv::Mat& overlap, cv::Mat& trajMap){

    std::unique_lock<std::mutex> lck(dataMutex);

    curPose = pose;

    if(mask.channels() == 3){
        curMask = mask.clone();
    }else{
        cv::cvtColor(mask, curMask, CV_GRAY2RGB);
    }

    overlapMask = overlap.clone();

    demoMap = trajMap.clone();

    float theta = pose.theta;

    Eigen::Matrix2f Rota;
    Rota<< cos(theta), -sin(theta), sin(theta), cos(theta);

    Eigen::Vector2f pixelCenter = Rota * agvCenter + pose.trans;

    cv::circle(demoMap, cv::Point(0.5*pixelCenter(1), 0.5*pixelCenter(0)), 10, cv::Scalar(0, 255, 0), -1);

    // cv::Vec3b color(0, 255, 255);
    // demoMap.at<cv::Vec3b> (pixelCenter(0), pixelCenter(1)) = color;

}

void Viewer::run() {

    pangolin::CreateWindowAndBind("AgvViewer", 1400, 700);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //定义按钮面板
    //新建按钮和选择框
    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,0.2);
    pangolin::Var<bool> menu("menu.test",true,true);

    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));


    //map
    pangolin::View& map_image = pangolin::Display("map")
      .SetBounds(0.0, 1.0 ,0.2, 0.7, 1.0f)
      .SetLock(pangolin::LockLeft, pangolin::LockBottom);

    //frame
    // float rate = 560.0f/340.0f;
    float rate = 560.0f/580.0f;
    pangolin::View& frame_image = pangolin::Display("frame")
      .SetBounds(0.65, 1.0, 0.7, 1.0, rate)
      .SetLock(pangolin::LockLeft, pangolin::LockBottom);

    pangolin::View& overlap_image = pangolin::Display("overlap")
      .SetBounds(0.0, 0.35, 0.7, 1.0, rate)
      .SetLock(pangolin::LockLeft, pangolin::LockBottom);


    //初始化
    pangolin::GlTexture mapTexture(mapCols, mapRows, GL_RGB,true,0,GL_BGR,GL_UNSIGNED_BYTE);
    pangolin::GlTexture frameTexture(frameCols, frameRows, GL_RGB,true,0,GL_BGR,GL_UNSIGNED_BYTE);
    pangolin::GlTexture overlapTexture(frameCols, frameRows, GL_RGB,true,0,GL_BGR,GL_UNSIGNED_BYTE);

    cout<<"[Viewer] init finished"<<endl;

    while (!pangolin::ShouldQuit() && !endSwitch) {

        //clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        // vis_display.Activate(vis_camera);

        std::unique_lock<std::mutex> lock(dataMutex);

        mapTexture.Upload(demoMap.data, GL_BGR,GL_UNSIGNED_BYTE);
        map_image.Activate();
        glColor3f(1.0,1.0,1.0);
        mapTexture.RenderToViewportFlipY();

        frameTexture.Upload(curMask.data, GL_BGR,GL_UNSIGNED_BYTE);
        frame_image.Activate();
        glColor3f(1.0,1.0,1.0);
        frameTexture.RenderToViewportFlipY();

        overlapTexture.Upload(overlapMask.data, GL_BGR,GL_UNSIGNED_BYTE);
        overlap_image.Activate();
        glColor3f(1.0,1.0,1.0);
        overlapTexture.RenderToViewportFlipY();

        pangolin::FinishFrame();
    }

}
