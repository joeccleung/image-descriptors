#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class BaseDescriptorGenerationPipeline
{
public:
    BaseDescriptorGenerationPipeline();
    virtual ~BaseDescriptorGenerationPipeline();

    virtual void setDebug(bool set);
    virtual void setNormalize(bool set);
    void setStartEndPatchIndex(int start, int end);

    virtual cv::Mat compute(cv::Mat patch, int index, std::shared_ptr<cv::FileStorage> debugFile) = 0;

protected:
    bool shouldDebug;
    bool shouldNormalize;

    int startPatch;
    int endPatch;
};