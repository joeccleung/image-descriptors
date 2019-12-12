#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp> // For SIFT

#include "DescriptorGenerator.h"
#include "DistanceCalculator.h"

#include "NSSDCalculation.h"
#include "NSSDPipeline.h"

#include "TSDescriptorGenerationPipeline.h"

#include "T2GradientFieldPipeline.h"
#include "S1SquareGridBilinearWeightedPipeline.h"

using namespace std;
using namespace cv;

const int NUMBER_OF_PATCH_PER_IMAGE = 256;

const int COMMAND_EXIT = 0;
const int COMMAND_EXTRACT_IMAGE_PATCHES_FROM_DATASET = 1;
const int COMMAND_SIFT_MATCHING = 2;
const int COMMAND_NSSD_MATCHING = 3;
const int COMMAND_T2A_S1_MATCHING = 4;

const int T2_Bin_Length = 16;         // 16px x 16px in each bin
const int T2_Bin_Count_Each_Side = 4; // 16 bins per patch

int ShowMainMenu()
{
    int command = -1;

    cout << "========== Main Menu ==========" << endl;
    cout << "(0) Exit program;" << endl;
    cout << "(1) Extract image patches from dataset;" << endl;
    cout << "(2) SIFT" << endl;
    cout << "(3) NSSD" << endl;
    cout << "(4) T2a-S1" << endl;
    cout << endl;
    cout << "Please select a command: ";
    cin >> command;

    return command;
}

void CommandExtractImagePatchesFromDataset(int patchSize)
{
    int indexOfFirstPatch = 0;
    cout << "Please provide the first index of patches to extract (source/patchesXXXX.bmp):";
    cin >> indexOfFirstPatch;
    int indexOfLastPatch = 0;
    cout << "Please provide the last index of patches to extract (source/patchesXXXX.bmp):";
    cin >> indexOfLastPatch;

    int fileNameCounter = indexOfFirstPatch * NUMBER_OF_PATCH_PER_IMAGE;

    stringstream ss;
    stringstream outputSS;
    for (int i = indexOfFirstPatch; i <= indexOfLastPatch; i++)
    {
        stringstream().swap(ss);
        ss << "source/patches" << setfill('0') << setw(4) << i << ".bmp";

        Mat source = imread(ss.str());

        int patchRow = source.rows / patchSize;
        int patchCol = source.cols / patchSize;

        cout << "Reading patches:" << ss.str() << endl;

        for (int y = 0; y < patchRow; y++)
        {
            int startRow = y * patchSize;

            for (int x = 0; x < patchCol; x++)
            {
                int startCol = x * patchSize;

                Mat patch(cv::Size(patchSize, patchSize), CV_8U);

                for (int r = 0; r < patchSize; r++)
                {
                    for (int c = 0; c < patchSize; c++)
                    {
                        patch.at<uchar>(r, c) = source.at<Vec3b>(startRow + r, startCol + c)[0]; // Since the image is a greyscale, we just need one channel
                    }
                }

                stringstream().swap(outputSS);
                outputSS << "patch/" << setfill('0') << setw(4) << fileNameCounter << ".png";
                imwrite(outputSS.str(), patch);
                fileNameCounter++;
            }
        }
    }

    cout << "Generated " << fileNameCounter - indexOfFirstPatch * NUMBER_OF_PATCH_PER_IMAGE << " of image patches" << endl;
    cout << endl;
}

int askUserStartingPatch()
{
    int starting = -1;

    while (starting < 0)
    {
        cout << "Please provide the starting patch index: ";
        cin >> starting;
    }

    return starting;
}

int askUserEndingPatch()
{
    int ending = -1;

    while (ending < 0)
    {
        cout << "Please provide the ending patch index: ";
        cin >> ending;
    }

    return ending;
}

void CommandSIFTMatching()
{
}

void CommandNSSDMatching(int patchSize)
{
    int starting = askUserStartingPatch();
    int ending = askUserEndingPatch();

    DescriptorGenerator generator;
    NSSDPipeline nssdPipeline;
    generator.Begin("NSSD.xml", "NSSD_Debug.xml", starting, ending, &nssdPipeline);

    DistanceCalculator calculator;
    NSSDCalculation nssdCalculation;
    calculator.calculate("NSSD.xml", "NSSD_Result.xml", starting, ending, &nssdCalculation);
}

int ShowT2AS1Menu()
{
    int command = -1;

    cout << "========== T2A + S1 Menu ==========" << endl;
    cout << "(0) Return to main menu;" << endl;
    cout << "(1) Generate descriptors from patches;" << endl;
    cout << "(2) Calculate Euclidean distance between patches;" << endl;
    cout << endl;
    cout << "Please select a command: ";
    cin >> command;

    return command;
}

/**
 * @brief Perform T2-S1 Descriptor Generation on selected patches
*/
void CommandT2AS1GenerateDescriptorsFromPatches()
{
    int starting = askUserStartingPatch();
    int ending = askUserEndingPatch();

    DescriptorGenerator generator;
    TSDescriptorGenerationPipeline tsPipeline;
    T2GradientFieldPipeline t2Pipeline;
    S1SquareGridBilinearWeightedPipeline s1Pipeline;
    tsPipeline.setTBlock(&t2Pipeline);
    tsPipeline.setSBlock(&s1Pipeline);
    generator.Begin("T2S1.xml", "T2Sq_Debug.xml", starting, ending, &tsPipeline);
}

/**
 * @brief This is the sketch goal to classify an image
 * 
 */
// void CommandT2AS1GenerateDescriptorsFromImage()
// {
//     string path("");

//     while (path.length() == 0)
//     {
//         cout << "Please provide path to image:";
//         cin >> path;
//     }

//     Mat img = imread(path);
//     if (img.data == NULL)
//     {
//         cout << "Fail to open file " << path << endl;
//         return;
//     }

//     // Stage 1: Pre-smoothing
//     GaussianBlur(img, img, Size(7, 7), 2.7);

//     // Stage 2: SIFT keypoints detection
//     Ptr<xfeatures2d::SIFT> SIFT = xfeatures2d::SIFT::create(100);
//     vector<KeyPoint> SIFTKeypoints;
//     SIFT->detect(img, SIFTKeypoints, noArray());
//     cout << "Number of SIFT keypoints " << SIFTKeypoints.size() << endl;

//     // Stage 3: Select the SIFT keypoints that can form 64x64 patch
//     for (int i = 0; i < SIFTKeypoints.size(); i++)
//     {
//         if (SIFTKeypoints[i].pt.x < 31 || SIFTKeypoints[i].pt.x >= img.cols - 31)
//         {
//             continue;
//         }

//         if (SIFTKeypoints[i].pt.y < 31 || SIFTKeypoints[i].pt.y >= img.rows - 31)
//         {
//             continue;
//         }
//     }
// }

void CommandT2AS1CalculateEuclideanDistanceBetweenPatches()
{
    int start = -1;
    int end = -1;

    while (start < 0)
    {
        cout << "Please input the starting index of the patches: ";
        cin >> start;
    }

    while (end < 0)
    {
        cout << "Please input the ending index of the patches: ";
        cin >> end;

        if (start >= end)
        {
            cout << "Ending index cannot be smaller than or equal to starting index" << endl;
            end = -1;
        }
    }

    // Load the descriptors from the XML
    cv::FileStorage desFS = cv::FileStorage("T2AS1.xml", cv::FileStorage::READ);
    vector<Mat> listOfDescriptors;

    for (int i = start; i <= end; i++)
    {
        Mat des;
        string desName = "T2S1_" + to_string(i);
        desFS[desName] >> des;
        listOfDescriptors.push_back(des);
    }
    desFS.release();

    // Read the ground truth table
    ifstream groundTruthFile;
    groundTruthFile.open("GroundTruth.txt");
    if (!groundTruthFile.is_open())
    {
        groundTruthFile.close();
        cout << "Cannot find GroundTruth.txt" << endl;
        return;
    }

    vector<int> groundTruth;
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

    // Open the output file
    ofstream csvFile;
    csvFile.open("T2S1_Result.csv");

    // Calculate the Euclidean Distance
    for (int a = start; a < end; a++)
    {
        for (int b = a + 1; b <= end; b++)
        {
            csvFile << a << "," << b << "," << (groundTruth[a - start] == groundTruth[b - start]) << "," << norm(listOfDescriptors[a], listOfDescriptors[b]) << endl; // norm default calculates Euclidean distance (L2)
        }

        cout << "Calculated the Euclidean Distance from " << a << endl;
    }

    cout << "Number of descriptor " << listOfDescriptors.size() << endl;

    csvFile.close();
}

void CommandT2AS1()
{
    int command = -1;

    while (command != 0)
    {
        command = ShowT2AS1Menu();

        switch (command)
        {
        case 0:
            return;

        case 1:
            CommandT2AS1GenerateDescriptorsFromPatches();
            break;

        // case 2:
        //     CommandT2AS1GenerateDescriptorsFromImage();
        //     break;
        case 2:
            CommandT2AS1CalculateEuclideanDistanceBetweenPatches();
            break;

        default:
            cout << "Unknown command " << command << endl;
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    const int PATCH_SIZE = 64;

    int command = -1;

    while (command != 0)
    {
        command = ShowMainMenu();

        switch (command)
        {
        case COMMAND_EXIT:
            break;

        case COMMAND_EXTRACT_IMAGE_PATCHES_FROM_DATASET:
            CommandExtractImagePatchesFromDataset(PATCH_SIZE);
            break;

        case COMMAND_SIFT_MATCHING:
            CommandSIFTMatching();
            break;

        case COMMAND_NSSD_MATCHING:
            CommandNSSDMatching(PATCH_SIZE);
            break;

        case COMMAND_T2A_S1_MATCHING:
            CommandT2AS1();
            break;
        }
    }

    return 0;
}