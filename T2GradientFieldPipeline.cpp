#include "T2GradientFieldPipeline.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

const int T2_VF_P_X = 0;
const int T2_VF_N_X = 1;
const int T2_VF_P_Y = 2;
const int T2_VF_N_Y = 3;

T2GradientFieldPipeline::T2GradientFieldPipeline()
{
}

T2GradientFieldPipeline::~T2GradientFieldPipeline()
{
}

cv::Mat T2GradientFieldPipeline::compute(cv::Mat patch, int index, std::shared_ptr<cv::FileStorage> debugFile)
{
    Mat vectorField(patch.cols, patch.rows, CV_16SC4, Vec4s(0, 0, 0, 0)); // Create Float64 4 channels for rectified gradient vectors

    short value = 0;
    // Horizontal Kernel [-1|0|1]
    for (int r = 0; r < patch.rows; r++)
    {
        // First column
        // We clamp the L.H.S pixel with the center pixel
        value = -1 * patch.at<uchar>(r, 0); // Input is grayscale, RGB channels are the same
        value += patch.at<uchar>(r, 1);

        vectorField.at<Vec4s>(r, 0)[T2_VF_P_X] = abs(value) + value;
        vectorField.at<Vec4s>(r, 0)[T2_VF_N_X] = abs(value) - value;

        for (int c = 1; c < patch.cols - 1; c++)
        {
            value = -1 * patch.at<uchar>(r, c - 1);
            value += patch.at<uchar>(r, c + 1);

            vectorField.at<Vec4s>(r, c)[T2_VF_P_X] = abs(value) + value;
            vectorField.at<Vec4s>(r, c)[T2_VF_N_X] = abs(value) - value;
        }

        // Last column
        // We clamp the R.H.S pixel with the center pixel
        value = -1 * patch.at<uchar>(r, patch.cols - 2); // Input is grayscale, RGB channels are the same
        value += patch.at<uchar>(r, patch.cols - 1);

        vectorField.at<Vec4s>(r, patch.cols - 1)[T2_VF_P_X] = abs(value) + value;
        vectorField.at<Vec4s>(r, patch.cols - 1)[T2_VF_N_X] = abs(value) - value;
    }

    // Vectical Kernel [-1|0|1]
    for (int c = 0; c < patch.cols; c++)
    {
        // First row
        // We clamp the top pixel with the center pixel
        value = -1 * patch.at<uchar>(0, c); // Input is grayscale, RGB channels are the same
        value += patch.at<uchar>(1, c);

        vectorField.at<Vec4s>(0, c)[T2_VF_P_Y] = abs(value) + value;
        vectorField.at<Vec4s>(0, c)[T2_VF_N_Y] = abs(value) - value;

        for (int r = 1; r < patch.rows - 1; r++)
        {
            value = -1 * patch.at<uchar>(r - 1, c);
            value += patch.at<uchar>(r + 1, c);

            vectorField.at<Vec4s>(r, c)[T2_VF_P_Y] = abs(value) + value;
            vectorField.at<Vec4s>(r, c)[T2_VF_N_Y] = abs(value) - value;
        }

        // Last row
        // We clamp the bottom pixel with the center pixel
        value = -1 * patch.at<uchar>(patch.rows - 2, c); // Input is grayscale, RGB channels are the same
        value += patch.at<uchar>(patch.rows - 1, c);

        vectorField.at<Vec4s>(patch.rows - 1, c)[T2_VF_P_Y] = abs(value) + value;
        vectorField.at<Vec4s>(patch.rows - 1, c)[T2_VF_N_Y] = abs(value) - value;
    }

    // Vector field output
    if (shouldDebug)
    {
        *debugFile << "T" + to_string(index) << vectorField; // Matrix name must be prefix with non-numberic characters
    }

    return vectorField;
}