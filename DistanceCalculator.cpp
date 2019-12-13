#include "DistanceCalculator.h"

#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

DistanceCalculator::DistanceCalculator()
{
}

DistanceCalculator::~DistanceCalculator()
{
}

void DistanceCalculator::calculate(std::string descriptorFileName, std::string outputFileName, int startPatch, int endPatch, BaseDistanceCalculation *calculation)
{
    vector<int> groundTruth = loadGroundTruthTable(startPatch, endPatch);

    vector<Mat> descriptors;
    loadDescriptor(descriptorFileName, startPatch, endPatch, descriptors);

    // Output prep
    ofstream outputFile;
    outputFile.open(outputFileName);
    
    for (int a = 0; a < descriptors.size() - 1; a++)
    {
        for (int b = a + 1; b < descriptors.size(); b++)
        {
            double distance = calculation->calculate(descriptors.at(a), descriptors.at(b));

            outputFile << a + startPatch << "," << b + startPatch << "," << (groundTruth[a] == groundTruth[b]) << "," << distance << endl;
            cout << "SSD between " << a + startPatch << " and " << b + startPatch << " = " << distance << endl;
        }
    }

    outputFile.close();
}

bool DistanceCalculator::loadDescriptor(std::string fileName, int start, int end, std::vector<cv::Mat> &descriptors) const
{
    descriptors.clear();

    // Load the descriptors from the XML
    cv::FileStorage desFS = cv::FileStorage(fileName, cv::FileStorage::READ);

    for (int i = start; i <= end; i++)
    {
        Mat des;
        string desName = "D_" + to_string(i);
        desFS[desName] >> des;
        descriptors.push_back(des);
    }
    desFS.release();

    return true;
}

std::vector<int> DistanceCalculator::loadGroundTruthTable(int start, int end) const
{
    vector<int> groundTruth;

    // Load Ground Truth
    ifstream groundTruthFile;
    groundTruthFile.open("GroundTruth.txt");
    if (!groundTruthFile.is_open())
    {
        groundTruthFile.close();
        cout << "Cannot find GroundTruth.txt" << endl;
        return groundTruth;
    }

    int kpTag;
    int _3DTag;
    // Seek to the desired starting location
    for (int s = 0; s < start; s++)
    {
        groundTruthFile.ignore(numeric_limits<streamsize>::max(), groundTruthFile.widen('\n'));
    }
    for (int e = start; e < end; e++)
    {
        if (groundTruthFile.eof())
        {
            break;
        }
        groundTruthFile >> kpTag >> _3DTag;
        groundTruth.push_back(kpTag);
    }
    groundTruthFile.close();
    cout << "Loaded the Ground Truth Table" << endl;

    return groundTruth;
}