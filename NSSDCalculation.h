#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "BaseDistanceCalculation.h"

class NSSDCalculation : public BaseDistanceCalculation
{
public:
    NSSDCalculation();
    virtual ~NSSDCalculation();

    double calculate(cv::Mat &desA, cv::Mat &desB);
};