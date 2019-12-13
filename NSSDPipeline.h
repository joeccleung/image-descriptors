#include "BaseDescriptorGenerationPipeline.h"

class NSSDPipeline : public BaseDescriptorGenerationPipeline
{
public:
    NSSDPipeline();
    virtual ~NSSDPipeline();

    cv::Mat compute(cv::Mat patch, int index, std::shared_ptr<cv::FileStorage> debugFile);
};