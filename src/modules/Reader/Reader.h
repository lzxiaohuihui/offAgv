
#include "../Base/Base.hpp"


using namespace std;

class Reader{

public:
    Reader();

    ~Reader();

    bool readPoseAndTicks();

    Pose getReferPose(double curTick, vector<Pose*>& poseList, int& start);

    string txt_path;

    vector<double> captureTicks; 

    bool withLidarPoses;
    bool withOdomPoses;

    vector<Pose*> lidarPoses;
    vector<Pose*> odomPoses;


private:

    bool readCapTicks();

    bool readLidarPoses();

    bool readOdomPoses();

    double lidarFps;


};