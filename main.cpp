#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp> // For SIFT

using namespace std;
using namespace cv;

const int COMMAND_EXIT = 0;
const int COMMAND_EXTRACT_IMAGE_PATCHES_FROM_DATASET = 1;
const int COMMAND_SIFT_MATCHING = 2;
const int COMMAND_NSSD_MATCHING = 3;
const int COMMAND_T2A_S1_MATCHING = 4;

struct ImagePatch_t
{
    Mat patch;
    vector<KeyPoint> keypoints;
    Mat descriptor;
};

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
    int indexOfLastPatch = 0;
    cout << "Please provide the last index of patches to extract (source/patchesXXXX.bmp):";
    cin >> indexOfLastPatch;

    int fileNameCounter = 0;

    stringstream ss;
    stringstream outputSS;
    for (int i = 0; i <= indexOfLastPatch; i++)
    {
        stringstream().swap(ss);
        ss << "source/patches" << setfill('0') << setw(4) << indexOfLastPatch << ".bmp";

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

                Mat patch(cv::Size(patchSize, patchSize), CV_8UC3);

                for (int r = 0; r < patchSize; r++)
                {
                    for (int c = 0; c < patchSize; c++)
                    {
                        patch.at<Vec3b>(r, c) = source.at<Vec3b>(startRow + r, startCol + c);
                    }
                }

                stringstream().swap(outputSS);
                outputSS << "patch/" << setfill('0') << setw(4) << fileNameCounter << ".png";
                imwrite(outputSS.str(), patch);
                fileNameCounter++;
            }
        }
    }

    cout << "Generated " << fileNameCounter << " of image patches" << endl;

    cout << "Close the window to continue" << endl;

    waitKey(0);
}

void CommandSIFTMatching()
{
    int numberOfPatches = 0;

    while (numberOfPatches < 2)
    {
        cout << "How many patches to match? " << endl;
        cin >> numberOfPatches;

        if (numberOfPatches < 2)
        {
            cout << "Number of patches to match must be greater than or equal to 2" << endl;
        }
    }

    vector<ImagePatch_t> patches;

    for (int i = 0; i < numberOfPatches; i++)
    {
        ImagePatch_t patchData;
        patchData.patch = imread(to_string(i) + ".png");
        patches.push_back(patchData);
    }

    Ptr<xfeatures2d::SIFT> SIFT = xfeatures2d::SIFT::create();

    for (int p = 0; p < numberOfPatches; p++)
    {
        SIFT->detectAndCompute(patches[p].patch, noArray(), patches[p].keypoints, patches[p].descriptor, false);
        cout << "SIFT detect and computed " << p << " patch" << endl;
    }

    // for (int p = 0; p < numberOfPatches - 1; p++)
    // {
    //     double total = 0;
    //     for (int r = 0; r < patches[p].descriptor.rows; r++)
    //     {
    //         for (int k = 0; k < patches[p + 1].descriptor.rows; k++)
    //         {
    //             total += sqrt(patches[p].descriptor.row(r).dot(patches[p + 1].descriptor.row(k)));
    //         }
    //     }

    //     double average = total / (patches[p].descriptor.rows * patches[p + 1].descriptor.rows);

    //     cout << "Distance between " << p << " and " << (p+1) << " is " << average << endl;
    // }

    vector<DMatch> matches;
    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    for (int p = 0; p < numberOfPatches - 1; p++)
    {
        for (int w = p + 1; w < numberOfPatches; w++)
        {
            matcher->match(patches[p].descriptor, patches[w].descriptor, matches);

            double total = 0;
            for (int m = 0; m < matches.size(); m++)
            {
                total += matches[m].distance;
            }

            double average = total / matches.size();

            // cout << average << endl;
            cout << "Distance between " << p << " and " << w << " = " << average << endl;
        }
    }
}

void CommandNSSDMatching(int patchSize)
{
    int numberOfPatches = 0;

    while (numberOfPatches < 2)
    {
        cout << "How many patches to match? " << endl;
        cin >> numberOfPatches;

        if (numberOfPatches < 2)
        {
            cout << "Number of patches to match must be greater than or equal to 2" << endl;
        }
    }

    vector<ImagePatch_t> patches;

    for (int i = 0; i < numberOfPatches; i++)
    {
        ImagePatch_t patchData;
        patchData.patch = imread(to_string(i) + ".png");
        patches.push_back(patchData);
    }

    // Calculate SSD between patch a and patch b (=a+1)
    for (int a = 0; a < patches.size() - 1; a++)
    {
        for (int b = a + 1; b < patches.size(); b++)
        {
            int ssd = 0;
            for (int r = 0; r < patchSize; r++)
            {
                for (int c = 0; c < patchSize; c++)
                {
                    int diff = patches[a].patch.at<Vec3b>(r, c)[0] - patches[b].patch.at<Vec3b>(r, c)[0]; // Since we are using greyscale image, we only need to compute one color channel
                    ssd += diff * diff;
                }
            }

            cout << "SSD between " << a << " and " << b << " = " << ssd << endl;
        }
    }
}

int ShowT2AS1Menu()
{
    int command = -1;

    cout << "========== T2A + S1 Menu ==========" << endl;
    cout << "(0) Return to main menu;" << endl;
    cout << "(1) Generate descriptors from image patches;" << endl;
    cout << "(2) Generate descriptors from image" << endl;
    cout << endl;
    cout << "Please select a command: ";
    cin >> command;

    return command;
}

void CommandT2AS1GenerateDescriptorsFromPatches()
{
    int numberOfPatches = -1;

    while (numberOfPatches < 0)
    {
        cout << "Please specify the last index of the patches: ";
        cin >> numberOfPatches;
    }

    // Output descriptors to file as XML
    cv::FileStorage fs("T2AS1.xml", FileStorage::WRITE);

    stringstream inputFileName;
    for (int i = 0; i <= numberOfPatches; i++)
    {
        stringstream().swap(inputFileName);
        inputFileName << "patch/" << setfill('0') << setw(4) << i << ".png";

        Mat img;
        img = imread(inputFileName.str());
        Mat vectorField(img.cols, img.rows, CV_64F, Scalar(0));

        // Horizontal Kernel
        fs << "M" + to_string(i) << vectorField;

        cout << "Processed " << inputFileName.str() << endl;
    }

    fs.release();
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

        case 2:

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