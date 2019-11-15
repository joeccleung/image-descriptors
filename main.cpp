#include <iostream>
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
    cout << "(2) SIFT Matching" << endl;
    cout << endl;
    cout << "Please select a command: ";
    cin >> command;

    return command;
}

void CommandExtractImagePatchesFromDataset(int patchSize)
{
    string pathToImages;

    cout << "Please provide path to images for patches extraction" << endl;
    cin >> pathToImages;

    Mat source = imread(pathToImages);

    int patchRow = source.rows / patchSize;
    int patchCol = source.cols / patchSize;

    int fileNameCounter = 0;

    cout << "Number of patches " << patchRow << " x " << patchCol << " = " << patchRow * patchCol << endl;

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

            imwrite(to_string(fileNameCounter) + ".png", patch);
            fileNameCounter++;
            cout << "Extracted " << fileNameCounter << endl;
        }
    }

    namedWindow("Test");
    imshow("Test", source);

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
        for (int w = p+1; w < numberOfPatches; w++)
        {
            matcher->match(patches[p].descriptor, patches[w].descriptor, matches);

            double total = 0;
            for (int m = 0; m < matches.size(); m++)
            {
                total += matches[m].distance;
            }

            double average = total / matches.size();

            cout << average << endl;
            // cout << "Distance between " << p << " and " << w << " = " << average << endl;
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
        }
    }

    return 0;
}