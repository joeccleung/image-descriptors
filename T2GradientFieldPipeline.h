#pragma once

#include "BaseDescriptorGenerationPipeline.h"

class T2GradientFieldPipeline : public BaseDescriptorGenerationPipeline
{
public:
    T2GradientFieldPipeline();
    virtual ~T2GradientFieldPipeline();

    cv::Mat compute(cv::Mat patch, int index, std::shared_ptr<cv::FileStorage> debugFile);
};