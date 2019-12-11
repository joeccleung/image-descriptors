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

const int NUMBER_OF_PATCH_PER_IMAGE = 256;

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

    cout << "Generated " << fileNameCounter - indexOfFirstPatch * NUMBER_OF_PATCH_PER_IMAGE << " of image patches" << endl;
    cout << endl;
}

void CommandSIFTMatching()
{

}

void CommandNSSDMatching(int patchSize)
{
    int starting = -1;
    int ending = -1;
    int numberOfPatches = 0;

    while (starting < 0)
    {
        cout << "Please provide the starting patch ";
        cin >> starting;
    }

    while (ending < 0)
    {
        cout << "Please provide the ending patch ";
        cin >> ending;
    }

    FileStorage debugFile = FileStorage("NSSD_Debug.xml", FileStorage::WRITE);

    stringstream inputFileName;
    vector<Mat> patches;
    // Stage 1: Load the patches
    for (int i = starting; i <= ending; i++)
    {
        stringstream().swap(inputFileName);
        inputFileName << "patch/" << setfill('0') << setw(4) << i << ".png";

        patches.push_back(imread(inputFileName.str()));
    }
    cout << "Loaded " << patches.size() << " patches" << endl;

    // TODO: Reduce the channel of the matrix to speed up the process

    vector<Mat> normalized;
    // Stage 2: Normalization
    for (int i = 0; i < patches.size(); i++)
    {
        debugFile << "Original_" + to_string(i) << patches[i];

        Mat normalizeMat(patches[i].size(), CV_64FC3);
        normalize(patches[i], normalizeMat, 1, 0, NORM_L2, CV_64FC3);
        normalized.push_back(normalizeMat);

        debugFile << "Normalize_" + to_string(i) << normalized[i];

        // Stage 3: Threshold to reduce dynamic range of the image
        min(normalized[i], 0.154, normalized[i]);

        debugFile << "K_" + to_string(i) << normalized[i];

        // Stage 4: Normalization again
        normalize(normalized[i], normalized[i], 1, NORM_L2);

        debugFile << "N_" + to_string(i) << normalized[i];
    }
    cout << "Normalization complete " << endl;

    // Output prep
    ofstream outputFile;
    outputFile.open("NSSD_Result.csv");

    // Stage 6: Load Ground Truth
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
    for (int s = 0; s < starting; s++)
    {
        groundTruthFile.ignore(numeric_limits<streamsize>::max(), groundTruthFile.widen('\n'));
    }
    for (int e = starting; e < ending; e++)
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

    // Stage 5: SSD
    for (int a = 0; a < normalized.size() - 1; a++)
    {
        for (int b = a + 1; b < normalized.size(); b++)
        {
            int ssd = 0;
            for (int r = 0; r < patchSize; r++)
            {
                for (int c = 0; c < patchSize; c++)
                {
                    int diff = normalized[a].at<Vec3b>(r, c)[0] - normalized[b].at<Vec3b>(r, c)[0]; // Since we are using greyscale image, we only need to compute one color channel
                    ssd += diff * diff;
                }
            }

            outputFile << a + starting << "," << b + starting << "," << (groundTruth[a] == groundTruth[b]) << "," << ssd << endl;
            cout << "SSD between " << a + starting << " and " << b + starting << " = " << ssd << endl;
        }
    }

    outputFile.close();
    debugFile.release();
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
    cv::FileStorage outFS("T2AS1.xml", FileStorage::WRITE);
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
            }
        }

        if (debug)
        {
            debugFS << "S" + to_string(i) << bins;
        }

        // Flatten the bins into 64 length descriptor
        outFS << "T2S1_" + to_string(i) << bins.reshape(1, 1);

        cout << "Processed " << inputFileName.str() << endl;
    }

    outFS.release();
    debugFS.release();
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
    string path("");

    while (path.length() == 0)
    {
        cout << "Please provide path to image:";
        cin >> path;
    }

    Mat img = imread(path);
    if (img.data == NULL)
    {
        cout << "Fail to open file " << path << endl;
        return;
    }

    // Stage 1: Pre-smoothing
    GaussianBlur(img, img, Size(7, 7), 2.7);

    // Stage 2: SIFT keypoints detection
    Ptr<xfeatures2d::SIFT> SIFT = xfeatures2d::SIFT::create(100);
    vector<KeyPoint> SIFTKeypoints;
    SIFT->detect(img, SIFTKeypoints, noArray());
    cout << "Number of SIFT keypoints " << SIFTKeypoints.size() << endl;

    // Stage 3: Select the SIFT keypoints that can form 64x64 patch
    for(int i = 0; i < SIFTKeypoints.size(); i++) 
    {
        if (SIFTKeypoints[i].pt.x < 31 || SIFTKeypoints[i].pt.x >= img.cols-31) {
            continue;
        }

        if (SIFTKeypoints[i].pt.y < 31 || SIFTKeypoints[i].pt.y >= img.rows-31) {
            continue;
        }
    }
}
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
            CommandT2AS1GenerateDescriptorsFromImage();
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