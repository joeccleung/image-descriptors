#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class BaseDistanceCalculation
{
public:
    BaseDistanceCalculation();
    virtual ~BaseDistanceCalculation();

    virtual double calculate(cv::Mat &desA, cv::Mat &desB) = 0;

};