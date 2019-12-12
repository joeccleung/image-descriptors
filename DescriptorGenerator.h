#pragma once

#include "BaseDescriptorGenerationPipeline.h"

class DescriptorGenerator
{
public:
    DescriptorGenerator();
    ~DescriptorGenerator();

    void Begin(std::string descriptorFileName, std::string debugFileName, int starting, int ending, BaseDescriptorGenerationPipeline* pipeline);
};