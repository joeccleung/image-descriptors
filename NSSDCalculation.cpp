#include "NSSDCalculation.h"

using namespace std;
using namespace cv;

NSSDCalculation::NSSDCalculation() : BaseDistanceCalculation()
{
}

NSSDCalculation::~NSSDCalculation()
{
}

double NSSDCalculation::calculate(cv::Mat &desA, cv::Mat &desB)
{
    int dim = desA.cols;

    double ssd = 0;
    for (int c = 0; c < dim; c++)
    {
        double diff = desA.at<double>() - desB.at<double>();
        ssd += diff * diff;
    }

    return ssd;
}
