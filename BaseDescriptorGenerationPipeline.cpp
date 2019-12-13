#include "BaseDescriptorGenerationPipeline.h"

#include <iostream>
#include <fstream>

using namespace std;

BaseDescriptorGenerationPipeline::BaseDescriptorGenerationPipeline()
    : shouldDebug(false),
      shouldNormalize(false),
      startPatch(-1),
      endPatch(-1)
{
}

BaseDescriptorGenerationPipeline::~BaseDescriptorGenerationPipeline()
{
}

void BaseDescriptorGenerationPipeline::setDebug(bool set)
{
    shouldDebug = set;
}

void BaseDescriptorGenerationPipeline::setNormalize(bool set)
{
    shouldNormalize = set;
}

void BaseDescriptorGenerationPipeline::setStartEndPatchIndex(int start, int end)
{
    startPatch = start;
    endPatch = end;
}

