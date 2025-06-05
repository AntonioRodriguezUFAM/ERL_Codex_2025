// AlgorithmConcreteKernels.cuh
#pragma once

// Cuda Library

#include "../../usr/local/cuda-10.2/include/cuda_runtime.h"

//#include "../../usr/local/cuda-10.2/include/device_launch_parameters.h"


#include "../SharedStructures/ZeroCopyFrameData.h"
#include "../SharedStructures/AlgorithmConfig.h"

// Custom clamp function for CUDA compatibility
__host__ __device__ inline int clamp(int value, int min, int max) {
    return value < min ? min : (value > max ? max : value);
}


// CUDA kernel declarations
__global__ void sobelEdgeKernel(const uint8_t* input, uint8_t* output, int width, int height);
__global__ void medianFilterKernel(const uint8_t* input, uint8_t* output, int width, int height, int windowSize);
__global__ void histogramEqualizationKernel(const uint8_t* input, uint8_t* output, int width, int height, const int* cdf, int minCdf);
__global__ void gaussianBlurVerticalKernel(const uint8_t* input, uint8_t* output, int width, int height, int radius);
__global__ void gaussianBlurHorizontalKernel(const uint8_t* input, uint8_t* output, int width, int height, int radius);

// Wrapper functions for kernel launches
namespace AlgorithmConcreteKernels {
    void launchSobelEdgeKernel(const std::shared_ptr<ZeroCopyFrameData>& frame, std::vector<uint8_t>& processedBuffer);
    void launchMedianFilterKernel(const std::shared_ptr<ZeroCopyFrameData>& frame, std::vector<uint8_t>& processedBuffer, int windowSize);
    void launchHistogramEqualizationKernel(const std::shared_ptr<ZeroCopyFrameData>& frame, std::vector<uint8_t>& processedBuffer);
    void launchHeterogeneousGaussianBlurKernel(const std::shared_ptr<ZeroCopyFrameData>& frame, std::vector<uint8_t>& processedBuffer, int radius);
}