//AlgorithmConcreteKernels.cu

// AlgorithmConcreteKernels.cu
#include "AlgorithmConcreteKernels.cuh"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include "../../usr/local/cuda-10.2/include/cuda_runtime.h"

//#include "../../usr/local/cuda-10.2/include/device_launch_parameters.h"

// // Custom clamp function for CUDA compatibility
// __host__ __device__ inline int clamp(int value, int min, int max) {
//     return value < min ? min : (value > max ? max : value);
// }


// CUDA kernel definitions
__global__ void sobelEdgeKernel(const uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int idx = (y * width + x) * 2;
        int gx = (-input[(y-1)*width*2+(x-1)*2] + input[(y-1)*width*2+(x+1)*2])
               + (-2*input[y*width*2+(x-1)*2] + 2*input[y*width*2+(x+1)*2])
               + (-input[(y+1)*width*2+(x-1)*2] + input[(y+1)*width*2+(x+1)*2]);
        int gy = (-input[(y-1)*width*2+(x-1)*2] - 2*input[(y-1)*width*2+x*2] - input[(y-1)*width*2+(x+1)*2])
               + (input[(y+1)*width*2+(x-1)*2] + 2*input[(y+1)*width*2+x*2] + input[(y+1)*width*2+(x+1)*2]);
        int mag = sqrtf(gx * gx + gy * gy);
        //output[idx] = std::min(255, std::max(0, mag));
        output[idx] = clamp(mag, 0, 255); // Replaced std::min(std::max(...))
        output[idx + 1] = 128;
    }
}

__global__ void medianFilterKernel(const uint8_t* input, uint8_t* output, int width, int height, int windowSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int halfWindow = windowSize / 2;
    if (x >= halfWindow && x < width - halfWindow && y >= halfWindow && y < height - halfWindow) {
        int idx = (y * width + x) * 2;
        uint8_t window[25]; // Assuming 5x5 window max
        int k = 0;
        for (int dy = -halfWindow; dy <= halfWindow; dy++) {
            for (int dx = -halfWindow; dx <= halfWindow; dx++) {
                window[k++] = input[(y + dy) * width * 2 + (x + dx) * 2];
            }
        }
        // Simple bubble sort for median
        for (int i = 0; i < k - 1; i++)
            for (int j = 0; j < k - i - 1; j++)
                if (window[j] > window[j + 1]) {
                    uint8_t temp = window[j];
                    window[j] = window[j + 1];
                    window[j + 1] = temp;
                }
        output[idx] = window[k / 2];
        output[idx + 1] = 128;
    }
}

__global__ void histogramEqualizationKernel(const uint8_t* input, uint8_t* output, int width, int height, const int* cdf, int minCdf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 2;
        int val = input[idx];
        output[idx] = ((cdf[val] - minCdf) * 255) / (width * height - minCdf);
        output[idx + 1] = 128;
    }
}

__global__ void gaussianBlurHorizontalKernel(const uint8_t* input, uint8_t* output, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 2;
        float sum = 0.f;
        float wsum = 0.f;
        for (int dx = -radius; dx <= radius; dx++) {
            //int xx = std::min(std::max(x + dx, 0), width - 1);
            int xx = clamp(x + dx, 0, width - 1); // Replaced std::min(std::max(...))
            float w = expf(-(dx * dx) / (2.f * radius * radius));
            sum += input[y * width * 2 + xx * 2] * w;
            wsum += w;
        }
        output[idx] = (uint8_t)(sum / wsum);
        output[idx + 1] = 128;
    }
}

__global__ void gaussianBlurVerticalKernel(const uint8_t* input, uint8_t* output, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 2;
        float sum = 0.f;
        float wsum = 0.f;
        for (int dy = -radius; dy <= radius; dy++) {
            //int yy = std::min(std::max(y + dy, 0), height - 1);
            int yy = clamp(y + dy, 0, height - 1); // Replaced std::min(std::max(...))
            float w = expf(-(dy * dy) / (2.f * radius * radius));
            sum += input[yy * width * 2 + x * 2] * w;
            wsum += w;
        }
        output[idx] = (uint8_t)(sum / wsum);
        output[idx + 1] = 128;
    }
}



// Wrapper function implementations
namespace AlgorithmConcreteKernels {
    void launchSobelEdgeKernel(const std::shared_ptr<ZeroCopyFrameData>& frame, std::vector<uint8_t>& processedBuffer) {
        if (!frame || !frame->dataPtr) {
            spdlog::error("[AlgorithmConcreteKernels] Invalid zero-copy input frame.");
            return;
        }
        uint8_t *d_input, *d_output;
        cudaError_t err;
        err = cudaMalloc((void**)&d_input, frame->size);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in SobelEdge input allocation: {}", cudaGetErrorString(err));
            return;
        }
        err = cudaMalloc((void**)&d_output, frame->size);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in SobelEdge output allocation: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            return;
        }
        err = cudaMemcpy(d_input, frame->dataPtr, frame->size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in SobelEdge input copy: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            cudaFree(d_output);
            return;
        }
        dim3 blockDim(16, 16);
        dim3 gridDim((frame->width + blockDim.x - 1) / blockDim.x, (frame->height + blockDim.y - 1) / blockDim.y);
        sobelEdgeKernel<<<gridDim, blockDim>>>(d_input, d_output, frame->width, frame->height);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in SobelEdge kernel execution: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            cudaFree(d_output);
            return;
        }
        processedBuffer.resize(frame->size);
        err = cudaMemcpy(processedBuffer.data(), d_output, frame->size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in SobelEdge output copy: {}", cudaGetErrorString(err));
        }
        cudaFree(d_input);
        cudaFree(d_output);
    }

    void launchMedianFilterKernel(const std::shared_ptr<ZeroCopyFrameData>& frame, std::vector<uint8_t>& processedBuffer, int windowSize) {
        if (!frame || !frame->dataPtr) {
            spdlog::error("[AlgorithmConcreteKernels] Invalid zero-copy input frame.");
            return;
        }
        uint8_t *d_input, *d_output;
        cudaError_t err;
        err = cudaMalloc((void**)&d_input, frame->size);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in MedianFilter input allocation: {}", cudaGetErrorString(err));
            return;
        }
        err = cudaMalloc((void**)&d_output, frame->size);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in MedianFilter output allocation: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            return;
        }
        err = cudaMemcpy(d_input, frame->dataPtr, frame->size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in MedianFilter input copy: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            cudaFree(d_output);
            return;
        }
        dim3 blockDim(16, 16);
        dim3 gridDim((frame->width + blockDim.x - 1) / blockDim.x, (frame->height + blockDim.y - 1) / blockDim.y);
        medianFilterKernel<<<gridDim, blockDim>>>(d_input, d_output, frame->width, frame->height, windowSize);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in MedianFilter kernel execution: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            cudaFree(d_output);
            return;
        }
        processedBuffer.resize(frame->size);
        err = cudaMemcpy(processedBuffer.data(), d_output, frame->size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in MedianFilter output copy: {}", cudaGetErrorString(err));
        }
        cudaFree(d_input);
        cudaFree(d_output);
    }

    void launchHistogramEqualizationKernel(const std::shared_ptr<ZeroCopyFrameData>& frame, std::vector<uint8_t>& processedBuffer) {
        if (!frame || !frame->dataPtr) {
            spdlog::error("[AlgorithmConcreteKernels] Invalid zero-copy input frame.");
            return;
        }
        std::vector<int> histogram(256, 0), cdf(256, 0);
        for (size_t i = 0; i < frame->size; i += 2) {
            histogram[static_cast<uint8_t*>(frame->dataPtr)[i]]++;
        }
        int minCdf = 0;
        for (int i = 0; i < 256; i++) {
            cdf[i] = (i == 0) ? histogram[i] : cdf[i - 1] + histogram[i];
            if (cdf[i] > 0 && minCdf == 0) minCdf = cdf[i];
        }
        uint8_t *d_input, *d_output;
        int *d_cdf;
        cudaError_t err;
        err = cudaMalloc((void**)&d_input, frame->size);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in Histogram input allocation: {}", cudaGetErrorString(err));
            return;
        }
        err = cudaMalloc((void**)&d_output, frame->size);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in Histogram output allocation: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            return;
        }
        err = cudaMalloc((void**)&d_cdf, 256 * sizeof(int));
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in Histogram CDF allocation: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            cudaFree(d_output);
            return;
        }
        err = cudaMemcpy(d_input, frame->dataPtr, frame->size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in Histogram input copy: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_cdf);
            return;
        }
        err = cudaMemcpy(d_cdf, cdf.data(), 256 * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in Histogram CDF copy: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_cdf);
            return;
        }
        dim3 blockDim(16, 16);
        dim3 gridDim((frame->width + blockDim.x - 1) / blockDim.x, (frame->height + blockDim.y - 1) / blockDim.y);
        histogramEqualizationKernel<<<gridDim, blockDim>>>(d_input, d_output, frame->width, frame->height, d_cdf, minCdf);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in Histogram kernel execution: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_cdf);
            return;
        }
        processedBuffer.resize(frame->size);
        err = cudaMemcpy(processedBuffer.data(), d_output, frame->size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in Histogram output copy: {}", cudaGetErrorString(err));
        }
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_cdf);
    }

    void launchHeterogeneousGaussianBlurKernel(const std::shared_ptr<ZeroCopyFrameData>& frame, std::vector<uint8_t>& processedBuffer, int radius) {
        if (!frame || !frame->dataPtr) {
            spdlog::error("[AlgorithmConcreteKernels] Invalid zero-copy input frame.");
            return;
        }
        const int width = frame->width, height = frame->height;
        std::vector<uint8_t> temp(frame->size);
        uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
        // CPU: Horizontal pass
        for (size_t y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sum = 0.f, wsum = 0.f;
                for (int dx = -radius; dx <= radius; dx++) {
                    int xx = clamp(x + dx, 0, width - 1);
                    float w = std::exp(-(dx * dx) / (2.f * radius * radius));
                    sum += data[y * width * 2 + xx * 2] * w;
                    wsum += w;
                }
                temp[y * width * 2 + x * 2] = (uint8_t)(sum / wsum);
                temp[y * width * 2 + x * 2 + 1] = 128;
            }
        }
        // GPU: Vertical pass
        uint8_t *d_input, *d_output;
        cudaError_t err;
        err = cudaMalloc((void**)&d_input, frame->size);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in HeterogeneousGaussianBlur input allocation: {}", cudaGetErrorString(err));
            return;
        }
        err = cudaMalloc((void**)&d_output, frame->size);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in HeterogeneousGaussianBlur output allocation: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            return;
        }
        err = cudaMemcpy(d_input, temp.data(), frame->size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in HeterogeneousGaussianBlur input copy: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            cudaFree(d_output);
            return;
        }
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
        gaussianBlurVerticalKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, radius);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in HeterogeneousGaussianBlur kernel execution: {}", cudaGetErrorString(err));
            cudaFree(d_input);
            cudaFree(d_output);
            return;
        }
        processedBuffer.resize(frame->size);
        err = cudaMemcpy(processedBuffer.data(), d_output, frame->size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            spdlog::error("[AlgorithmConcreteKernels] CUDA error in HeterogeneousGaussianBlur output copy: {}", cudaGetErrorString(err));
        }
        cudaFree(d_input);
        cudaFree(d_output);
    }
}



//========================================================================


// //===================== CUDA =============================================
// //  MOD: Implemented CUDA kernel for Sobel edge detection
// // CUDA kernels
// __global__ void sobelEdgeKernel(const uint8_t* input, uint8_t* output, int width, int height) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
//         int idx = (y * width + x) * 2;
//         int gx = (-input[(y-1)*width*2+(x-1)*2] + input[(y-1)*width*2+(x+1)*2])
//                + (-2*input[y*width*2+(x-1)*2] + 2*input[y*width*2+(x+1)*2])
//                + (-input[(y+1)*width*2+(x-1)*2] + input[(y+1)*width*2+(x+1)*2]);
//         int gy = (-input[(y-1)*width*2+(x-1)*2] - 2*input[(y-1)*width*2+x*2] - input[(y-1)*width*2+(x+1)*2])
//                + (input[(y+1)*width*2+(x-1)*2] + 2*input[(y+1)*width*2+x*2] + input[(y+1)*width*2+(x+1)*2]);
//         int mag = sqrtf(gx * gx + gy * gy);
//         output[idx] = std::min(255, std::max(0, mag));
//         output[idx + 1] = 128;
//     }
// }


// __global__ void medianFilterKernel(const uint8_t* input, uint8_t* output, int width, int height, int windowSize) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int halfWindow = windowSize / 2;
//     if (x >= halfWindow && x < width - halfWindow && y >= halfWindow && y < height - halfWindow) {
//         int idx = (y * width + x) * 2;
//         uint8_t window[25]; // Assuming 5x5 window max
//         int k = 0;
//         for (int dy = -halfWindow; dy <= halfWindow; dy++) {
//             for (int dx = -halfWindow; dx <= halfWindow; dx++) {
//                 window[k++] = input[(y + dy) * width * 2 + (x + dx) * 2];
//             }
//         }
//         // Simple bubble sort for median (small window size)
//         for (int i = 0; i < k - 1; i++)
//             for (int j = 0; j < k - i - 1; j++)
//                 if (window[j] > window[j + 1]) {
//                     uint8_t temp = window[j];
//                     window[j] = window[j + 1];
//                     window[j + 1] = temp;
//                 }
//         output[idx] = window[k / 2];
//         output[idx + 1] = 128;
//     }
// }

// __global__ void histogramEqualizationKernel(const uint8_t* input, uint8_t* output, int width, int height, const int* cdf, int minCdf) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x < width && y < height) {
//         int idx = (y * width + x) * 2;
//         int val = input[idx];
//         output[idx] = ((cdf[val] - minCdf) * 255) / (width * height - minCdf);
//         output[idx + 1] = 128;
//     }
// }

// __global__ void gaussianBlurVerticalKernel(const uint8_t* input, uint8_t* output, int width, int height, int radius) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x < width && y < height) {
//         int idx = (y * width + x) * 2;
//         float sum = 0.f;
//         float wsum = 0.f;
//         for (int dy = -radius; dy <= radius; dy++) {
//             int yy = std::min(std::max(y + dy, 0), height - 1);
//             float w = expf(-(dy * dy) / (2.f * radius * radius));
//             sum += input[yy * width * 2 + x * 2] * w;
//             wsum += w;
//         }
//         output[idx] = (uint8_t)(sum / wsum);
//         output[idx + 1] = 128;
//     }
// }
// __global__ void gaussianBlurHorizontalKernel(const uint8_t* input, uint8_t* output, int width, int height, int radius) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x < width && y < height) {
//         int idx = (y * width + x) * 2;
//         float sum = 0.f;
//         float wsum = 0.f;
//         for (int dx = -radius; dx <= radius; dx++) {
//             int xx = std::min(std::max(x + dx, 0), width - 1);
//             float w = expf(-(dx * dx) / (2.f * radius * radius));
//             sum += input[y * width * 2 + xx * 2] * w;
//             wsum += w;
//         }
//         output[idx] = (uint8_t)(sum / wsum);
//         output[idx + 1] = 128;
//     }
// }

// // MOD: Implemented CUDA kernel for Gaussian blur

// // Process CUDA filters


// inline void AlgorithmConcrete::processSobelEdgeZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
//     if (!frame || !frame->dataPtr) {
//         spdlog::error("[AlgorithmConcrete] Invalid zero-copy input frame.");
//         return;
//     }
//     uint8_t *d_input, *d_output;
//     allocateCudaMemory((void**)&d_input, frame->size, "SobelEdge input allocation");
//     allocateCudaMemory((void**)&d_output, frame->size, "SobelEdge output allocation");
//     checkCudaError(cudaMemcpy(d_input, frame->dataPtr, frame->size, cudaMemcpyHostToDevice), "SobelEdge input copy");
//     dim3 blockDim(16, 16);
//     dim3 gridDim((frame->width + blockDim.x - 1) / blockDim.x, (frame->height + blockDim.y - 1) / blockDim.y);
//     sobelEdgeKernel<<<gridDim, blockDim>>>(d_input, d_output, frame->width, frame->height);
//     checkCudaError(cudaDeviceSynchronize(), "SobelEdge kernel execution");
//     processedBuffer_.resize(frame->size);
//     checkCudaError(cudaMemcpy(processedBuffer_.data(), d_output, frame->size, cudaMemcpyDeviceToHost), "SobelEdge output copy");
//     freeCudaMemory(d_input, "SobelEdge input free");
//     freeCudaMemory(d_output, "SobelEdge output free");
// }

// inline void AlgorithmConcrete::processMedianFilterZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
//     if (!frame || !frame->dataPtr) {
//         spdlog::error("[AlgorithmConcrete] Invalid zero-copy input frame.");
//         return;
//     }
//     uint8_t *d_input, *d_output;
//     allocateCudaMemory((void**)&d_input, frame->size, "MedianFilter input allocation");
//     allocateCudaMemory((void**)&d_output, frame->size, "MedianFilter output allocation");
//     checkCudaError(cudaMemcpy(d_input, frame->dataPtr, frame->size, cudaMemcpyHostToDevice), "MedianFilter input copy");
//     dim3 blockDim(16, 16);
//     dim3 gridDim((frame->width + blockDim.x - 1) / blockDim.x, (frame->height + blockDim.y - 1) / blockDim.y);
//     medianFilterKernel<<<gridDim, blockDim>>>(d_input, d_output, frame->width, frame->height, algoConfig_.medianWindowSize);
//     checkCudaError(cudaDeviceSynchronize(), "MedianFilter kernel execution");
//     processedBuffer_.resize(frame->size);
//     checkCudaError(cudaMemcpy(processedBuffer_.data(), d_output, frame->size, cudaMemcpyDeviceToHost), "MedianFilter output copy");
//     freeCudaMemory(d_input, "MedianFilter input free");
//     freeCudaMemory(d_output, "MedianFilter output free");
// }

// inline void AlgorithmConcrete::processHistogramEqualizationZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
//     if (!frame || !frame->dataPtr) {
//         spdlog::error("[AlgorithmConcrete] Invalid zero-copy input frame.");
//         return;
//     }
//     uint8_t *d_input, *d_output;
//     int *d_histogram, *d_cdf;
//     std::vector<int> histogram(256, 0), cdf(256, 0);
//     for (size_t i = 0; i < frame->size; i += 2)
//         histogram[static_cast<uint8_t*>(frame->dataPtr)[i]]++;
//     int minCdf = 0;
//     for (int i = 0; i < 256; i++) {
//         cdf[i] = (i == 0) ? histogram[i] : cdf[i - 1] + histogram[i];
//         if (cdf[i] > 0 && minCdf == 0) minCdf = cdf[i];
//     }
//     allocateCudaMemory((void**)&d_input, frame->size, "Histogram input allocation");
//     allocateCudaMemory((void**)&d_output, frame->size, "Histogram output allocation");
//     allocateCudaMemory((void**)&d_cdf, 256 * sizeof(int), "Histogram CDF allocation");
//     checkCudaError(cudaMemcpy(d_input, frame->dataPtr, frame->size, cudaMemcpyHostToDevice), "Histogram input copy");
//     checkCudaError(cudaMemcpy(d_cdf, cdf.data(), 256 * sizeof(int), cudaMemcpyHostToDevice), "Histogram CDF copy");
//     dim3 blockDim(16, 16);
//     dim3 gridDim((frame->width + blockDim.x - 1) / blockDim.x, (frame->height + blockDim.y - 1) / blockDim.y);
//     histogramEqualizationKernel<<<gridDim, blockDim>>>(d_input, d_output, frame->width, frame->height, d_cdf, minCdf);
//     checkCudaError(cudaDeviceSynchronize(), "Histogram kernel execution");
//     processedBuffer_.resize(frame->size);
//     checkCudaError(cudaMemcpy(processedBuffer_.data(), d_output, frame->size, cudaMemcpyDeviceToHost), "Histogram output copy");
//     freeCudaMemory(d_input, "Histogram input free");
//     freeCudaMemory(d_output, "Histogram output free");
//     freeCudaMemory(d_cdf, "Histogram CDF free");
// }

// inline void AlgorithmConcrete::processHeterogeneousGaussianBlurZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
//     const int radius = algoConfig_.blurRadius, width = frame->width, height = frame->height;
//     processedBuffer_.resize(frame->size);
//     std::vector<uint8_t> temp(frame.size);
//     uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
//     // CPU: Horizontal pass
//     parallelFor(0, height, [&](size_t y) {
//         for (int x = 0; x < width; x++) {
//             float sum = 0.f, wsum = 0.f;
//             for (int dx = -radius; dx <= radius; dx++) {
//                 int xx = std::clamp(x + dx, 0, width - 1);
//                 float w = std::exp(-(dx * dx) / (2.f * radius * radius));
//                 sum += data[y * width * 2 + xx * 2] * w;
//                 wsum += w;
//             }
//             temp[y * width * 2 + x * 2] = (uint8_t)(sum / wsum);
//             temp[y * width * 2 + x * 2 + 1] = 128;
//         }
//     });
//     // GPU: Vertical pass
//     uint8_t *d_input, *d_output;
//     allocateCudaMemory((void**)&d_input, frame->size, "HeterogeneousGaussianBlur input allocation");
//     allocateCudaMemory((void**)&d_output, frame->size, "HeterogeneousGaussianBlur output allocation");
//     checkCudaError(cudaMemcpy(d_input, temp.data(), frame->size, cudaMemcpyHostToDevice), "HeterogeneousGaussianBlur input copy");
//     dim3 blockDim(16, 16);
//     dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
//     gaussianBlurVerticalKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, radius);
//     checkCudaError(cudaDeviceSynchronize(), "HeterogeneousGaussianBlur kernel execution");
//     processedBuffer_.resize(frame->size);
//     checkCudaError(cudaMemcpy(processedBuffer_.data(), d_output, frame->size, cudaMemcpyDeviceToHost), "HeterogeneousGaussianBlur output copy");
//     freeCudaMemory(d_input, "HeterogeneousGaussianBlur input free");
//     freeCudaMemory(d_output, "HeterogeneousGaussianBlur output free");
// }
