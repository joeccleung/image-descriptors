#pragma once

#include "BaseDescriptorGenerationPipeline.h"

class TSDescriptorGenerationPipeline : public BaseDescriptorGenerationPipeline
{
public:
    TSDescriptorGenerationPipeline();
    virtual ~TSDescriptorGenerationPipeline();

    virtual void setDebug(bool set);
    virtual void setNormalize(bool set);

    void setTBlock(BaseDescriptorGenerationPipeline *tblock);
    void setSBlock(BaseDescriptorGenerationPipeline *sblock);

    cv::Mat compute(cv::Mat patch, int index, std::shared_ptr<cv::FileStorage> debugFile);

private:
    BaseDescriptorGenerationPipeline *tblock;
    BaseDescriptorGenerationPipeline *sblock;
};