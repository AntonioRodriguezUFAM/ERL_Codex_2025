
// LucasKanadeOpticalFlow.h
#pragma once

#include "../SharedStructures/ZeroCopyFrameData.h"
#include "../SharedStructures/OpticalFlowConfig.h"

#include "../Others/LKof_defines.h"
#include <vector>
#include <cstring>
#include <complex>

class LucasKanadeOpticalFlow {
public:
    explicit LucasKanadeOpticalFlow(const OpticalFlowConfig& config);

    bool computeOpticalFlow(
        const std::shared_ptr<ZeroCopyFrameData>& prevFrame,
        const std::shared_ptr<ZeroCopyFrameData>& nextFrame,
        std::shared_ptr<ZeroCopyFrameData>& outputFlowFrame);

private:
    OpticalFlowConfig config_;

    bool matrixInversion(float A[2][2], float B[2], float& Vx, float& Vy);
    void isotropicFilter(const uint8_t* input, uint8_t* output, int width, int height);
    void computeMotionVectors(
        const uint8_t* img1, const uint8_t* img2,
        std::vector<short>& vx, std::vector<short>& vy,
        int width, int height);
};

// LucasKanadeOpticalFlow.cpp
//#include "LucasKanadeOpticalFlow.h"


LucasKanadeOpticalFlow::LucasKanadeOpticalFlow(const OpticalFlowConfig& config)
    : config_(config) {}

bool LucasKanadeOpticalFlow::matrixInversion(float A[2][2], float B[2], float& Vx, float& Vy) {
    float det = (A[0][0] * A[1][1]) - (A[0][1] * A[1][0]);
    if (std::abs(det) < config_.threshold) return false;
    

    float inv_det = 1.0f / det;
    Vx = inv_det * (A[1][1] * (-B[0]) - A[0][1] * (-B[1]));
    Vy = inv_det * (-A[1][0] * (-B[0]) + A[0][0] * (-B[1]));
    return true;
}

void LucasKanadeOpticalFlow::isotropicFilter(const uint8_t* input, uint8_t* output, int width, int height) {
    // Apply isotropic filter (5x5 smoothing kernel as per Xilinx reference)
   // const int halfFilter = FILTER_SIZE / 2; // Use FILTER_SIZE from LKof_defines.h
    //const int FILTER_SIZE = 5;
    const int offset = FILTER_SIZE / 2;

    for (int y = offset; y < height - offset; ++y) {
        for (int x = offset; x < width - offset; ++x) {
            int accum = 0;
            for (int dy = -offset; dy <= offset; ++dy)
                for (int dx = -offset; dx <= offset; ++dx)
                    accum += input[(y + dy) * width + (x + dx)];

            output[y * width + x] = static_cast<uint8_t>(accum / (FILTER_SIZE * FILTER_SIZE));
        }
    }
}

void LucasKanadeOpticalFlow::computeMotionVectors(
    const uint8_t* img1, const uint8_t* img2,
    std::vector<short>& vx, std::vector<short>& vy,
    int width, int height) {
    // Compute basic spatial and temporal derivatives Ix, Iy, It.
    // Simplified example without complex HLS optimization:
    std::vector<float> Ix(width * height, 0);
    std::vector<float> Iy(width * height, 0);
    std::vector<float> It(width * height, 0);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            Ix[idx] = (img1[idx + 1] - img1[idx - 1]) / 2.0f;
            Iy[idx] = (img1[idx + width] - img1[idx - width]) / 2.0f;
            It[idx] = img2[idx] - img1[idx];
        }
    }

    // Solve LK equations per pixel
    int winSize = config_.windowSize;
    int wOffset = winSize / 2;

    for (int y = wOffset; y < height - wOffset; ++y) {
        for (int x = wOffset; x < width - wOffset; ++x) {
            float sumIx2 = 0, sumIy2 = 0, sumIxIy = 0, sumIxIt = 0, sumIyIt = 0;

            for (int dy = -wOffset; dy <= wOffset; ++dy) {
                for (int dx = -wOffset; dx <= wOffset; ++dx) {
                    int idx = (y + dy) * width + (x + dx);
                    float ix = Ix[idx], iy = Iy[idx], it = It[idx];
                    sumIx2 += ix * ix;
                    sumIy2 += iy * iy;
                    sumIxIy += ix * iy;
                    sumIxIt += ix * it;
                    sumIyIt += iy * it;
                }
            }

            float A[2][2] = {{sumIx2, sumIxIy}, {sumIxIy, sumIy2}};
            float B[2] = {sumIxIt, sumIyIt};
            float vx_f, vy_f;

            if (matrixInversion(A, B, vx_f, vy_f)) {
                vx[y * width + x] = static_cast<short>(vx_f * 8);
                vy[y * width + x] = static_cast<short>(vy_f * 8);
            } else {
                vx[y * width + x] = vy[y * width + x] = 0;
            }
        }
    }
}

bool LucasKanadeOpticalFlow::computeOpticalFlow(
    const std::shared_ptr<ZeroCopyFrameData>& prevFrame,
    const std::shared_ptr<ZeroCopyFrameData>& nextFrame,
    std::shared_ptr<ZeroCopyFrameData>& outputFlowFrame) {

    int width = prevFrame->width;
    int height = prevFrame->height;

    outputFlowFrame = std::make_shared<ZeroCopyFrameData>();
    outputFlowFrame->width = width;
    outputFlowFrame->height = height;
    outputFlowFrame->size = prevFrame->size;
    outputFlowFrame->dataPtr = malloc(prevFrame->size);

    std::vector<short> vx(width * height, 0);
    std::vector<short> vy(width * height, 0);

    computeMotionVectors(
        static_cast<uint8_t*>(prevFrame->dataPtr),
        static_cast<uint8_t*>(nextFrame->dataPtr),
        vx, vy, width, height);

    memcpy(outputFlowFrame->dataPtr, vx.data(), prevFrame->size / 2);
    memcpy(static_cast<uint8_t*>(outputFlowFrame->dataPtr) + (prevFrame->size / 2), vy.data(), prevFrame->size / 2);

    return true;
}
