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

const int T2_VF_P_X = 0;
const int T2_VF_N_X = 1;
const int T2_VF_P_Y = 2;
const int T2_VF_N_Y = 3;
const int T2_Bin_Length = 16;         // 16px x 16px in each bin
const int T2_Bin_Count_Each_Side = 4; // 16 bins per patch

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
    char printDebugCommand = ' ';
    bool debug = false;

    while (numberOfPatches < 0)
    {
        cout << "Please specify the last index of the patches: ";
        cin >> numberOfPatches;
    }

    while (printDebugCommand == ' ')
    {
        cout << "Do you want intermediate output for debug? (y/n):";
        cin >> printDebugCommand;
    }

    debug = (printDebugCommand == 'y' || printDebugCommand == 'Y');

    // Output descriptors to file as XML
    cv::FileStorage debugFS("T2AS1_Debug.xml", FileStorage::WRITE);

    stringstream inputFileName;
    for (int i = 0; i <= numberOfPatches; i++)
    {
        stringstream().swap(inputFileName);
        inputFileName << "patch/" << setfill('0') << setw(4) << i << ".png";

        Mat img;
        img = imread(inputFileName.str());
        Mat vectorField(img.cols, img.rows, CV_16SC4, Vec4s(0, 0, 0, 0)); // Create Float64 4 channels for rectified gradient vectors

        short value = 0;
        // Horizontal Kernel [-1|0|1]
        for (int r = 0; r < img.rows; r++)
        {
            // First column
            // We clamp the L.H.S pixel with the center pixel
            value = -1 * img.at<Vec3b>(r, 0)[0]; // Input is grayscale, RGB channels are the same
            value += img.at<Vec3b>(r, 1)[0];

            vectorField.at<Vec4s>(r, 0)[T2_VF_P_X] = (value > 0) ? value : 0;
            vectorField.at<Vec4s>(r, 0)[T2_VF_N_X] = (value < 0) ? -1 * value : 0;

            for (int c = 1; c < img.cols - 1; c++)
            {
                value = -1 * img.at<Vec3b>(r, c - 1)[0];
                value += img.at<Vec3b>(r, c + 1)[0];

                vectorField.at<Vec4s>(r, c)[T2_VF_P_X] = (value > 0) ? value : 0;
                vectorField.at<Vec4s>(r, c)[T2_VF_N_X] = (value < 0) ? -1 * value : 0;
            }

            // Last column
            // We clamp the R.H.S pixel with the center pixel
            value = -1 * img.at<Vec3b>(r, img.cols - 2)[0]; // Input is grayscale, RGB channels are the same
            value += img.at<Vec3b>(r, img.cols - 1)[0];

            vectorField.at<Vec4s>(r, img.cols - 1)[T2_VF_P_X] = (value > 0) ? value : 0;
            vectorField.at<Vec4s>(r, img.cols - 1)[T2_VF_N_X] = (value < 0) ? -1 * value : 0;
        }

        // Vectical Kernel [-1|0|1]
        for (int c = 0; c < img.cols; c++)
        {
            // First row
            // We clamp the top pixel with the center pixel
            value = -1 * img.at<Vec3b>(0, c)[0]; // Input is grayscale, RGB channels are the same
            value += img.at<Vec3b>(1, c)[0];

            vectorField.at<Vec4s>(0, c)[T2_VF_P_Y] = (value > 0) ? value : 0;
            vectorField.at<Vec4s>(0, c)[T2_VF_N_Y] = (value < 0) ? -1 * value : 0;

            for (int r = 1; r < img.rows - 1; r++)
            {
                value = -1 * img.at<Vec3b>(r - 1, c)[0];
                value += img.at<Vec3b>(r + 1, c)[0];

                vectorField.at<Vec4s>(r, c)[T2_VF_P_Y] = (value > 0) ? value : 0;
                vectorField.at<Vec4s>(r, c)[T2_VF_N_Y] = (value < 0) ? -1 * value : 0;
            }

            // Last row
            // We clamp the bottom pixel with the center pixel
            value = -1 * img.at<Vec3b>(img.rows - 2, c)[0]; // Input is grayscale, RGB channels are the same
            value += img.at<Vec3b>(img.rows - 1, c)[0];

            vectorField.at<Vec4s>(img.rows - 1, c)[T2_VF_P_Y] = (value > 0) ? value : 0;
            vectorField.at<Vec4s>(img.rows - 1, c)[T2_VF_N_Y] = (value < 0) ? -1 * value : 0;
        }

        // Vector field output
        if (debug)
        {
            debugFS << "T" + to_string(i) << vectorField; // Matrix name must be prefix with non-numberic characters
        }

        // Binning
        Mat bins(T2_Bin_Count_Each_Side, T2_Bin_Count_Each_Side, CV_64FC4, Vec4d(0, 0, 0, 0));

        // Bilinear weighting
        Mat bilinear = (Mat_<double>(4, 4) << 0.015625, 0.046875, 0.046875, 0.015625, 0.046875, 0.1406, 0.1406, 0.046875, 0.046875, 0.1406, 0.1406, 0.046875, 0.015625, 0.046875, 0.046875, 0.015625);

        for (int y = 0; y < T2_Bin_Count_Each_Side; y++)
        {
            for (int x = 0; x < T2_Bin_Count_Each_Side; x++)
            {
                Vec4d total(0, 0, 0, 0);
                for (int r = 0; r < T2_Bin_Length; r++)
                {
                    for (int c = 0; c < T2_Bin_Length; c++)
                    {
                        total += vectorField.at<Vec4s>(y * T2_Bin_Length + r, x * T2_Bin_Length + c);
                    }
                }

                bins.at<Vec4d>(y, x) = total * bilinear.at<double>(y, x);
                cout << y << " " << x << " " << total << " " << bilinear.at<double>(y, x) << " " << (total * bilinear.at<double>(y, x)) << endl;
            }
        }

        if (debug)
        {
            debugFS << "S" + to_string(i) << bins;
        }

        cout << "Processed " << inputFileName.str() << endl;
    }

    debugFS.release();
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