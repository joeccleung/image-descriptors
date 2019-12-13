#include "S1SquareGridBilinearWeightedPipeline.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

const int T2_Bin_Length = 16;         // 16px x 16px in each bin
const int T2_Bin_Count_Each_Side = 4; // 16 bins per patch
const double THRESHOLD = 0.154;

S1SquareGridBilinearWeightedPipeline::S1SquareGridBilinearWeightedPipeline()
{
}

S1SquareGridBilinearWeightedPipeline::~S1SquareGridBilinearWeightedPipeline()
{
}

cv::Mat S1SquareGridBilinearWeightedPipeline::compute(cv::Mat patch, int index, std::shared_ptr<cv::FileStorage> debugFile)
{
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
                    total += patch.at<Vec4s>(y * T2_Bin_Length + r, x * T2_Bin_Length + c);
                }
            }

            bins.at<Vec4d>(y, x) = total * bilinear.at<double>(y, x);
        }
    }

    // Flatten the bins into 64 length descriptor
    bins = bins.reshape(1, 1);

    if (shouldDebug)
    {
        *debugFile << "S" + to_string(index) << bins;
    }

    // Post Normalization
    if (shouldNormalize)
    {
        // Convert to unit vector
        normalize(bins, bins, 1, NORM_L2);

        if (shouldDebug)
        {
            *debugFile << "U" + to_string(index) << bins;
        }

        // Threshold
        min(bins, THRESHOLD, bins);

        if (shouldDebug)
        {
            *debugFile << "K" + to_string(index) << bins;
        }

        // Convert to unit vector again
        normalize(bins, bins, 1, NORM_L2);

        if (shouldDebug)
        {
            *debugFile << "N" + to_string(index) << bins;
        }
    }

    cout << "Processed S1 for patch " << index << endl;

    // Output
    return bins;
}