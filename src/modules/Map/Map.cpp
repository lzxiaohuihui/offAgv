#include "Map.h"

#include "../Base/ImgProcess.h"

using namespace std;

Map::Map() : inited(false) {}

Map::~Map() {}

bool Map::loadAndInit(const std::string &map_name, int pyramid_level_)
{

    if (pyramid_level_ <= 0 || pyramid_level_ > MAX_PYRAMID_LEVEL)
    {

        cout << "[Map::loadAndInit] invalid param pyramid_level " << pyramid_level_ << endl;
        return false;
    }

    rawData = cv::imread(map_name);

    if (rawData.empty())
    {

        cout << "[Map::loadAndInit] read map file error " << map_name << endl;
        return false;
    }

    if (rawData.type() != CV_8UC3)
    {

        cout << "[Map::loadAndInit] unexpected map format (expected 8UC3)" << endl;
        return false;
    }

    pyramid_level = pyramid_level_;
    int validRows = (rawData.rows >> (pyramid_level - 1)) << (pyramid_level - 1);
    int validCols = (rawData.cols >> (pyramid_level - 1)) << (pyramid_level - 1);

    data.create(validRows, validCols, CV_32FC1);
    for (int i = 0; i < data.rows; i++)
    {

        cv::Vec3b *rowPtr = rawData.ptr<cv::Vec3b>(i);

        for (int j = 0; j < data.cols; j++)
        {

            int pixel = rowPtr[j].val[0] + rowPtr[j].val[1] + rowPtr[j].val[2];

            data.at<float>(i, j) = (pixel > 30) ? 1.0f : 0.0f;
        }
    }

    cout << "[Map::loadAndInit] map crop to (row, col): (" << data.rows << ", " << data.cols << ")" << endl;
    // getCsmMap();
    mapList[0] = data;
    for (int l = 1; l < pyramid_level; l++)
    {

        if (mapList[l - 1].rows % 2 != 0 || mapList[l - 1].cols % 2 != 0)
        {

            cout << "[Map::loadAndInit] unexpected map rows/cols" << endl;
            return false;
        }

        mapList[l].create(mapList[l - 1].rows / 2, mapList[l - 1].cols / 2, CV_32F);
        ImgProcess::imageResizeHalf(mapList[l - 1], mapList[l]);
    }

    for (int l = 0; l < pyramid_level; l++)
    {

        cv::GaussianBlur(mapList[l], mapList[l],
                         cv::Size(gaussSize[l], gaussSize[l]), gaussSigma[l], gaussSigma[l]);
                         
        cv::Mat ima = mapList[l] * 255.0f;
        ima.convertTo(ima, CV_8UC1);
        string imaName = "map_" + to_string(l) + ".png";
        cv::imwrite(imaName, ima);

        double maxPixel;
        cv::minMaxIdx(mapList[l], NULL, &maxPixel, NULL, NULL);

        mapList[l] *= (float)(1.0 / maxPixel);

        // cv::Mat ima = mapList[l]*255.0f;
        // ima.convertTo(ima, CV_8UC1);
        // string imaName = "gaussMap_" + to_string(l) + ".png";
        // cv::imwrite(imaName, ima);

        gradList[l].create(mapList[l].rows, mapList[l].cols, CV_32FC2); // double channels
        ImgProcess::computeGradientDiff(mapList[l], gradList[l]);

        // *********icp
        // ********map grad
        cv::Mat icpGauss;
        cv::GaussianBlur(data, icpGauss, cv::Size(5, 5), 5, 5);

        icpGradMap.create(icpGauss.rows, icpGauss.cols, CV_32FC2);
        for (int i = 10; i < icpGauss.rows - 10; ++i)
        {
            Float2 *dstPtr = icpGradMap.ptr<Float2>(i);
            for (int j = 10; j < icpGauss.cols - 10; ++j)
            {
                dstPtr[j].x = icpGauss.at<float>(i + 1, j) - icpGauss.at<float>(i - 1, j);
                dstPtr[j].y = icpGauss.at<float>(i, j + 1) - icpGauss.at<float>(i, j - 1);
            }
        }
    }

    cout << "[Map::loadAndInit] init successful" << endl;

    inited = true;
    return true;
}

bool Map::getCsmMap()
{
    // 第0层地图
    csmList[0] = mapList[0];
    int block = 10;
    int rows = mapList[0].rows;
    int cols = mapList[0].cols;
    for (int i = 1; i < RESOLUTION_LEVEL; ++i)
    {
        int block_rows = rows / block;
        int block_cols = cols / block;
        csmList[i].create(rows, cols, mapList[0].type());
        for (int block_x = 0; block_x < block_rows; ++block_x)
        {
            for (int block_y = 0; block_y < block_cols; ++block_y)
            {

                float maxVal = 0.0f;
                for (int x = 0; x < block; ++x)
                {
                    for (int y = 0; y < block; ++y)
                    {
                        float curVal = csmList[i - 1].at<float>(block_x * block_rows + x, block_y * block_cols + y);
                        maxVal = curVal > maxVal ? curVal : maxVal;
                    }
                }
                for (int x = 0; x < block; ++x)
                {
                    for (int y = 0; y < block; ++y)
                    {
                        csmList[i].at<float>(block_x * block_rows + x, block_y * block_cols + y) = maxVal;
                    }
                }
            }
        }

        block *= 10;
    }
    cv::imwrite("csmMap.png", csmList[0]);

    return true;
}
