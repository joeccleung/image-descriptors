#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

const int COMMAND_EXIT = 0;
const int COMMAND_EXTRACT_IMAGE_PATCHES_FROM_DATASET = 1;

int ShowMainMenu()
{
    int command = -1;

    cout << "========== Main Menu ==========" << endl;
    cout << "(0) Exit program;" << endl;
    cout << "(1) Extract image patches from dataset;" << endl;
    cout << endl;
    cout << "Please select a command: ";
    cin >> command;

    return command;
}

void ExtractImagePatchesFromDataset(int patchSize)
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
            ExtractImagePatchesFromDataset(PATCH_SIZE);
            break;
        }
    }

    return 0;
}