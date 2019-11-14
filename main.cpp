#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    const int PATCH_SIZE = 64;
    string pathToImages;

    cout << "Please provide path to images for patches extraction" << endl;
    cin >> pathToImages;

    Mat source = imread(pathToImages);

    int patchRow = source.rows / PATCH_SIZE;
    int patchCol = source.cols / PATCH_SIZE;

    int fileNameCounter = 0;

    cout << "Number of patches " << patchRow << " x " << patchCol << " = " << patchRow * patchCol << endl;

    for (int y = 0; y < patchRow; y++)
    {
        int startRow = y * PATCH_SIZE;

        for (int x = 0; x < patchCol; x++)
        {
            int startCol = x * PATCH_SIZE;

            Mat patch(cv::Size(PATCH_SIZE, PATCH_SIZE), CV_8UC3);

            for (int r = 0; r < PATCH_SIZE; r++)
            {
                for (int c = 0; c < PATCH_SIZE; c++)
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

    waitKey(0);

    return 0;
}