#include "NSSDPipeline.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

NSSDPipeline::NSSDPipeline() : BaseDescriptorGenerationPipeline()
{
}

NSSDPipeline::~NSSDPipeline()
{
}

Mat NSSDPipeline::compute(Mat patch, int index, shared_ptr<cv::FileStorage> debugFile)
{
    if (shouldDebug)
    {
        *debugFile << "Original_" + to_string(index) << patch;
    }

    // Stage 1: Normalization
    Mat normalizeMat(patch.size(), CV_64F); // The normalized version of the patch
    normalize(patch, normalizeMat, 1, 0, NORM_L2, CV_64F);

    if (shouldDebug)
    {
        *debugFile << "Normalize_" + to_string(index) << normalizeMat;
    }

    // Stage 2: Threshold to reduce dynamic range of the image
    min(normalizeMat, 0.154, normalizeMat);

    if (shouldDebug)
    {
        *debugFile << "K_" + to_string(index) << normalizeMat;
    }

    // Stage 3: Normalization again
    normalize(normalizeMat, normalizeMat, 1, NORM_L2);

    if (shouldDebug)
    {
        *debugFile << "N_" + to_string(index) << normalizeMat;
    }

    // Stage 4: Flatten the descriptor into 1 x 4096 vector

    cout << "Normalization complete for " << index << endl;

    return normalizeMat.reshape(1, 1);
}