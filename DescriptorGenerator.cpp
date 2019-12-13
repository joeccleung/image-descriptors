#include "DescriptorGenerator.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

DescriptorGenerator::DescriptorGenerator()
{
}

DescriptorGenerator::~DescriptorGenerator()
{
}

void DescriptorGenerator::Begin(std::string descriptorFileName, std::string debugFileName, int starting, int ending, BaseDescriptorGenerationPipeline *pipeline)
{
    char shouldNormalizeCommand = ' ';
    bool shouldNormalize = false;

    char printDebugCommand = ' ';
    bool shouldDebug = false;

    pipeline->setStartEndPatchIndex(starting, ending);

    while (shouldNormalizeCommand == ' ')
    {
        cout << "Should I normalize the descriptor? (y/n): ";
        cin >> shouldNormalizeCommand;
    }
    shouldNormalize = (shouldNormalizeCommand == 'y' || shouldNormalizeCommand == 'Y');
    pipeline->setNormalize(shouldNormalize);

    while (printDebugCommand == ' ')
    {
        cout << "Do you want intermediate output for debug? (y/n):";
        cin >> printDebugCommand;
    }
    shouldDebug = (printDebugCommand == 'y' || printDebugCommand == 'Y');
    pipeline->setDebug(shouldDebug);
    shared_ptr<FileStorage> debugFile = make_shared<FileStorage>(FileStorage(debugFileName, FileStorage::WRITE));

    FileStorage outputFile = FileStorage(descriptorFileName, FileStorage::WRITE);
    stringstream inputFileName;
    // Stage 1: Load the patches
    for (int i = starting; i <= ending; i++)
    {
        stringstream().swap(inputFileName);
        inputFileName << "patch/" << setfill('0') << setw(4) << i << ".png";

        Mat patch = imread(inputFileName.str(), IMREAD_ANYDEPTH);
        Mat descriptor = pipeline->compute(patch, i, debugFile);

        outputFile << "D_" + to_string(i) << descriptor;
    }

    debugFile->release();
    outputFile.release();
}