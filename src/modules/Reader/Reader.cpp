
#include "Reader.h"

using namespace std;

Reader::Reader():
            lidarFps(15.0),
            withLidarPoses(false),
            withOdomPoses(false){

}

Reader::~Reader(){

}

bool Reader::readPoseAndTicks(){

    if(!readCapTicks()){
        cout<<"[Reader] failed to read 'capture_time.txt'"<<endl;
        return false;
    }

    if(!readLidarPoses()){
        cout<<"[Reader] failed to read 'lidar.txt'"<<endl;
    }

    if(!readOdomPoses()){
        cout<<"[Reader] failed to read 'odom.txt'"<<endl;
    }

    return true;

}

bool Reader::readCapTicks(){

    captureTicks.clear();

    string fileName = txt_path + "capture_time.txt";

    ifstream file;
    file.open(fileName);

    double tick;
    while(file >> tick){

        captureTicks.push_back(tick);
    }

    file.close();

    if(captureTicks.size() < 2)
        return false;

    return true;

}

bool Reader::readLidarPoses(){

    lidarPoses.clear();

    string fileName = txt_path + "lidar.txt";

    ifstream file;
    file.open(fileName);

    if(!file.is_open())
        return false;

    string str;
    file >> str;

    double localTick, AgvTick;
    file >> localTick;
    file >> AgvTick;
    
    double difTick = localTick - AgvTick -7.13;

    float transX, transY, theta;
    double tick;

    double lidarStep = 1.0 / lidarFps;
    // double lidarStep = 0.0;
 
    while(true){

        if(!((file >> transX) && (file >> transY) && (file >> theta) && (file >> tick)))
            break;

        Pose* pPose = new Pose(transX, transY, theta, tick - lidarStep + difTick);

        //cout<<transX<<" "<<transY<<" "<<theta<<" "<<tick<<endl;getchar();
        
        lidarPoses.push_back(pPose);
    }

    file.close();

    if(lidarPoses.size() < 2)
        return false;

    withLidarPoses = true;
    return true;

}

bool Reader::readOdomPoses(){

    odomPoses.clear();

    string fileName = txt_path + "odom.txt";

    ifstream file;
    file.open(fileName);

    if(!file.is_open())
        return false;

    string str;
    file >> str;

    double localTick, AgvTick;
    file >> localTick;
    file >> AgvTick;
    
    double difTick = localTick - AgvTick;

    float transX, transY, theta;
    double tick;

    double lidar_step = 0.06; 
    // double lidar_step = 0.0; 
    while(true){

        if(!((file >> transX) && (file >> transY) && (file >> theta) && (file >> tick)))
            break;

        Pose* pPose = new Pose(transX, transY, theta, tick - lidar_step + difTick);
        
        odomPoses.push_back(pPose);
    }

    file.close();

    if(odomPoses.size() < 2)
        return false;

    withOdomPoses = true;

    return true;

}


Pose Reader::getReferPose(double curTick, vector<Pose*>& poseList, int& start){

    vector<Pose*>::iterator iter = poseList.begin() + start;

    if((*iter)->tick > curTick)
        return *(*iter);

    int count = 0;
    while((*iter)->tick < curTick && iter < (poseList.end()-1)){

        iter++;
        count++;
    }
        
    double dist1 = abs((*(iter-1))->tick - curTick);
    double dist2 = abs((*iter)->tick - curTick);

    Pose* pPose = (dist1 < dist2) ? (*(iter-1)) : (*iter);

    start += count;

    return *pPose;
}