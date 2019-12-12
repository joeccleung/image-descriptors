#pragma once

#include "BaseDistanceCalculation.h"

#include <string>
#include <vector>

class DistanceCalculator
{
public:
    DistanceCalculator();
    virtual ~DistanceCalculator();

    void calculate(std::string descriptorFileName, std::string outputFileName, int startPatch, int endPatch, BaseDistanceCalculation *calculation);

private:
    bool loadDescriptor(std::string fileName, int start, int end, std::vector<cv::Mat> &descriptors) const;
    std::vector<int> loadGroundTruthTable(int start, int end) const;
};