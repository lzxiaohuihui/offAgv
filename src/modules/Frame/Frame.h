 #pragma once

#include "../Base/Base.hpp"

using namespace std;

class Frame {

public:

    Frame(){
    }

    ~Frame(){}

    bool init(int row, int col, int pyramid_level_);

    void buildFrame(string imaName);

	bool linesFit();

	bool degenerateCheck();

	bool correctCheck(float degeTheta);

	float getDeltaTheta(float theta1, float theta2);

    static const int MAX_PYRAMID_LEVEL = 3;

	// Images
	cv::Mat rawMask;
	cv::Mat edgeImage;
	
	std::vector<cv::Mat> maskList;

	std::vector<cv::Mat> gaussMaskList;
	std::vector<cv::Mat> gradMaskList;

	int pyramid_level;

	int rows[MAX_PYRAMID_LEVEL];
	int cols[MAX_PYRAMID_LEVEL];

    const float gaussSigma[3] = {5.0f , 2.5f, 1.0f};
    const int gaussSize[3] = {5, 3, 1};

	std::vector<LaneLine*> laneLines;

	// Results
	Pose pose;
	Pose rawPose;
	// Pose* smooth_pose;

	bool lost;
	bool degenerative;

	float degeTheta;

	int goodCorrectCount;
	
};