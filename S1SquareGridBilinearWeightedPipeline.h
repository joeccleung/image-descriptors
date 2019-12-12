#pragma once

#include "BaseDescriptorGenerationPipeline.h"

class S1SquareGridBilinearWeightedPipeline : public BaseDescriptorGenerationPipeline
{
public:
    S1SquareGridBilinearWeightedPipeline();
    virtual ~S1SquareGridBilinearWeightedPipeline();

    cv::Mat compute(cv::Mat patch, int index, std::shared_ptr<cv::FileStorage> debugFile);
};