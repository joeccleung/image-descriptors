#include "TSDescriptorGenerationPipeline.h"

#include <iostream>

using namespace std;
using namespace cv;

TSDescriptorGenerationPipeline::TSDescriptorGenerationPipeline() : BaseDescriptorGenerationPipeline(),
                                                                   tblock(nullptr),
                                                                   sblock(nullptr)
{
}

TSDescriptorGenerationPipeline::~TSDescriptorGenerationPipeline()
{
}

void TSDescriptorGenerationPipeline::setDebug(bool set)
{
    this->tblock->setDebug(set);
    this->sblock->setDebug(set);
}

void TSDescriptorGenerationPipeline::setNormalize(bool set)
{
    this->tblock->setNormalize(set);
    this->sblock->setNormalize(set);
}

void TSDescriptorGenerationPipeline::setTBlock(BaseDescriptorGenerationPipeline *tblock)
{
    this->tblock = tblock;
}

void TSDescriptorGenerationPipeline::setSBlock(BaseDescriptorGenerationPipeline *sblock)
{
    this->sblock = sblock;
}

cv::Mat TSDescriptorGenerationPipeline::compute(cv::Mat patch, int index, std::shared_ptr<cv::FileStorage> debugFile)
{
    if (tblock == nullptr || sblock == nullptr)
    {
        cout << "Did not specify T block or S block. Terminate descriptor generation " << endl;
        return Mat(); // Return an empty CV::Mat
    }

    Mat tblockResult = tblock->compute(patch, index, debugFile);
    Mat sblockResult = sblock->compute(tblockResult, index, debugFile);
    //TODO: Normalization

    return sblockResult;
}