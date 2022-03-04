
#include"./AgvTracker/AgvTracker.h"

#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

using namespace cv;

int main(int argc, char** argv){

    // mAgvTracker.map_name = "../data/507map_20.png";
    // mAgvTracker.data_path = "../data/data_test/";
    // pAgvTracker->map_name = "../dege_data/dege_map_10.png";
    // pAgvTracker->data_path = "../dege_data/data4/";


    AgvTracker* pAgvTracker = new AgvTracker();
    // pAgvTracker->map_name = "../dege_data/dege_map_10.png";
    // pAgvTracker->data_path = "../dege_data/data4/";
    pAgvTracker->map_name = "../data/507map_20.png";
    pAgvTracker->data_path = "../data/data_test/";

    // pAgvTracker->map_name = "/media/lzh/0E88335488333993/jingData/f5map_global3.png";
    // pAgvTracker->data_path = "/media/lzh/0E88335488333993/jingData/data_test1/";

    pAgvTracker->initTracker();

    // Viewer* pViewer = new Viewer();
    // pAgvTracker->pViewer = pViewer;

    cout<<"Init finished, press enter to continue"<<endl;
    // getchar();

    // thread* pThViewer = new thread(&Viewer::run, pViewer);

    pAgvTracker->run();
    
    // pThViewer->join();


    return 0;
}