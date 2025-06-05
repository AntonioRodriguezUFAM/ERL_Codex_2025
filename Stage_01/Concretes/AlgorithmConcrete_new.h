
// //=================== Woring final version

// // AlgorithmConcrete_new.h
// // (Stage 3: Algorithm Implementation)


// // AlgorithmConcrete_new.h
// // AlgorithmConcrete_new.h


// #pragma once

// #include "../Interfaces/IAlgorithm.h"
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/ZeroCopyFrameData.h"
// #include "../SharedStructures/SharedQueue.h"
// #include "../SharedStructures/ThreadManager.h"
// #include "../../Stage_02/Logger/PerformanceLogger.h"
// #include "../SharedStructures/AlgorithmConfig.h"

// #include <vector>
// #include <mutex>
// #include <atomic>
// #include <functional>
// #include <spdlog/spdlog.h>
// #include <chrono>
// #include <future>
// #include <cmath>

// class AlgorithmConcrete : public IAlgorithm {
// public:
//     explicit AlgorithmConcrete(ThreadManager& threadManager);

//     AlgorithmConcrete(std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
//         std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> outputQueue,
//         ThreadManager& threadManager);

//     AlgorithmConcrete(std::shared_ptr<SharedQueue<FrameData>> inputQueue,
//                      std::shared_ptr<SharedQueue<FrameData>> outputQueue,
//                      ThreadManager& threadManager);



//     ~AlgorithmConcrete() override;

//     static std::shared_ptr<IAlgorithm> createAlgorithm(
//         AlgorithmType type,
//         std::shared_ptr<SharedQueue<FrameData>> inputQueue,
//         std::shared_ptr<SharedQueue<FrameData>> outputQueue,
//         ThreadManager& threadManager);
//     static std::shared_ptr<IAlgorithm> createAlgorithmZeroCopy(
//         AlgorithmType type,
//         std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
//         std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> outputQueue,
//         ThreadManager& threadManager);

//     void startAlgorithm() override;
//     void stopAlgorithm() override;
//     bool processFrame(const FrameData& inputFrame, FrameData& outputFrame) override;
    
//     bool processFrameZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& inputFrame,
//                             std::shared_ptr<ZeroCopyFrameData>& outputFrame) override;

//     bool configure(const AlgorithmConfig& config) override;
//     void setErrorCallback(std::function<void(const std::string&)> callback) override;
//     std::tuple<double, double> getAlgorithmMetrics() const override;
//     double getLastFPS() const override;
//     double getFps() const override;
//     double getAverageProcTime() const override;
//     const uint8_t* getProcessedBuffer() const override;

// private:
//     void threadLoop();
//     void threadLoopZeroCopy();
//     void updateMetrics(double elapsedSec);
//     std::string algorithmTypeToString(AlgorithmType type) const;
//     void reportError(const std::string& msg);
//     void parallelFor(size_t start, size_t end, std::function<void(size_t)> func);

//     // Filter functions for ZeroCopyFrameData
//     // void filterGrayscale(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height);
//     // void filterInvert(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height);
//     // void filterSepia(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height);
//     // void filterEdge(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height);

//     // FrameData processing methods
//     void processInvert(const FrameData& frame);
//     void processGrayscale(const FrameData& frame);
//     void processEdgeDetection(const FrameData& frame);
//     void processGaussianBlur(const FrameData& frame);
//     void processMatrixMultiply();
//     void processMandelbrot();
//     void processPasswordHash(const FrameData& frame);
//     void processMultiPipeline(const FrameData& frame);
//     void processGPUMatrixMultiply();
//     void processMultiThreadedInvert(const FrameData& frame);

//     // ZeroCopyFrameData processing methods
//     void processInvertZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
//     void processGrayscaleZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
//     void processEdgeDetectionZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
//     void processGaussianBlurZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
//     void processMatrixMultiplyZeroCopy();
//     void processMandelbrotZeroCopy();
//     void processMultiPipelineZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
//     void processGPUMatrixMultiplyZeroCopy();
//     void processMultiThreadedInvertZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
//     void processPasswordHashZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);

//     std::atomic<bool> running_{false};
//     std::thread algoThread_;

//     std::shared_ptr<SharedQueue<FrameData>> inputQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> outputQueue_;

//     // std::shared_ptr<SharedQueue<ZeroCopyFrameData>> inputQueueZeroCopy_;
//     // std::shared_ptr<SharedQueue<ZeroCopyFrameData>> outputQueueZeroCopy_;

//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueueZeroCopy_;
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> outputQueueZeroCopy_;

//     std::vector<std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>> outputQueuesZeroCopy_; // Multiple output queues for filters

//     AlgorithmConfig algoConfig_;
//     std::function<void(const std::string&)> errorCallback_;

//     mutable std::mutex metricMutex_;
//     double fps_ = 0.0;
//     double avgProcTime_ = 0.0;
//     int framesCount_ = 0;
//     double totalProcTime_ = 0.0;

//     std::vector<uint8_t> processedBuffer_;

//     mutable std::mutex fpsMutex_;
//     mutable std::atomic<int> framesProcessed_{0};
//     mutable std::chrono::steady_clock::time_point lastUpdateTime_;
//     mutable double lastFPS_ = 0.0;
//     mutable double lastProcessingTime_ = 0.0;

//     ThreadManager& threadManager_;
// };

// // =============Implementation=====================================================
// inline AlgorithmConcrete::AlgorithmConcrete(ThreadManager& threadManager)
//     : running_{false}
//     , inputQueue_(nullptr)
//     , outputQueue_(nullptr)
//     , inputQueueZeroCopy_(nullptr)
//     , outputQueueZeroCopy_(nullptr)
//     , lastUpdateTime_(std::chrono::steady_clock::now())
//     , threadManager_(threadManager)
// {
//     spdlog::warn("[AlgorithmConcrete] Default constructor called - no queues provided.");
// }


// //=============================================================================================//
// inline AlgorithmConcrete::AlgorithmConcrete(std::shared_ptr<SharedQueue<FrameData>> inputQueue,
//     std::shared_ptr<SharedQueue<FrameData>> outputQueue,
//     ThreadManager& threadManager)
// : running_{false}
// , inputQueue_(std::move(inputQueue))
// , outputQueue_(std::move(outputQueue))
// , inputQueueZeroCopy_(nullptr)
// , outputQueueZeroCopy_(nullptr)
// , lastUpdateTime_(std::chrono::steady_clock::now())
// , threadManager_(threadManager)
// {
// spdlog::info("[AlgorithmConcrete] FrameData constructor called.");
// }

// //=============================================================================================//
// inline AlgorithmConcrete::AlgorithmConcrete(std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
//                                             std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> outputQueue,
//                                             ThreadManager& threadManager)
//     : running_{false}
//     , inputQueue_(nullptr)
//     , outputQueue_(nullptr)
//     , inputQueueZeroCopy_(std::move(inputQueue))
//     , outputQueueZeroCopy_(std::move(outputQueue))
//     , lastUpdateTime_(std::chrono::steady_clock::now())
//     , threadManager_(threadManager)
// {
//     spdlog::info("[AlgorithmConcrete] ZeroCopy constructor called.");
// }

// //=============================================================================================//

// // inline AlgorithmConcrete::AlgorithmConcrete(std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
// //                                             std::vector<std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>> outputQueues,
// //                                             ThreadManager& threadManager)
// //                                             : running_{false}
// //                                             , inputQueue_(nullptr)
// //                                             , outputQueue_(nullptr)
// //                                             , inputQueueZeroCopy_(std::move(inputQueue))
// //                                             , outputQueuesZeroCopy_(std::move(outputQueues))
// //                                             , lastUpdateTime_(std::chrono::steady_clock::now())
// //                                             , threadManager_(threadManager)
// //                                         {
// //                                             spdlog::info("[AlgorithmConcrete] ZeroCopy constructor called with multiple output queues.");
// //                                         }

// inline AlgorithmConcrete::~AlgorithmConcrete() {
//     stopAlgorithm();
// }

// inline std::shared_ptr<IAlgorithm> AlgorithmConcrete::createAlgorithm(
//     AlgorithmType type,
//     std::shared_ptr<SharedQueue<FrameData>> inputQueue,
//     std::shared_ptr<SharedQueue<FrameData>> outputQueue,
//     ThreadManager& threadManager)
// {
//     if (!inputQueue || !outputQueue) {
//         spdlog::error("[AlgorithmConcrete] FrameData queues must be non-null.");
//         return nullptr;
//     }
//     auto algo = std::make_shared<AlgorithmConcrete>(inputQueue, outputQueue, threadManager);
//     AlgorithmConfig cfg;
//     cfg.algorithmType = type;
//     algo->configure(cfg);
//     return algo;
// }

// inline std::shared_ptr<IAlgorithm> AlgorithmConcrete::createAlgorithmZeroCopy(
//     AlgorithmType type,
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> outputQueue,
//     ThreadManager& threadManager)
// {
//     if (!inputQueue || !outputQueue) {
//         spdlog::error("[AlgorithmConcrete] ZeroCopy queues must be non-null.");
//         return nullptr;
//     }
//     auto algo = std::make_shared<AlgorithmConcrete>(inputQueue, outputQueue, threadManager);
//     AlgorithmConfig cfg;
//     cfg.algorithmType = type;
//     algo->configure(cfg);
//     return algo;
// }

// inline void AlgorithmConcrete::startAlgorithm() {
//     if (running_) {
//         spdlog::warn("[AlgorithmConcrete] Algorithm is already running.");
//         return;
//     }
    
//     running_ = true;
//     if (inputQueueZeroCopy_ && outputQueueZeroCopy_) {
//         threadManager_.addThread("AlgorithmProcessingZeroCopy",
//             std::thread(&AlgorithmConcrete::threadLoopZeroCopy, this));
//     } else if (inputQueue_ && outputQueue_) {
//         threadManager_.addThread("AlgorithmProcessing",
//             std::thread(&AlgorithmConcrete::threadLoop, this));
//     } else {
//         spdlog::error("[AlgorithmConcrete] No valid queues initialized.");
//         running_ = false;
//         return;
//     }
    
//     spdlog::info("[AlgorithmConcrete] Algorithm thread started ({}).",
//                  algorithmTypeToString(algoConfig_.algorithmType));
// }

// inline void AlgorithmConcrete::stopAlgorithm() {
//     if (!running_) return;
//     running_ = false;
//     if (outputQueue_) {
//         outputQueue_->stop();
//     }
//     if (outputQueueZeroCopy_) {
//         outputQueueZeroCopy_->stop();
//     }
//     threadManager_.joinThreadsFor("AlgorithmProcessing");
//     threadManager_.joinThreadsFor("AlgorithmProcessingZeroCopy");
//     spdlog::info("[AlgorithmConcrete] Algorithm thread stopped.");
// }

// inline bool AlgorithmConcrete::configure(const AlgorithmConfig& config) {
//     algoConfig_ = config;
//     spdlog::info("[AlgorithmConcrete] Configured algorithm: {}", 
//                  algorithmTypeToString(config.algorithmType));
//     return true;
// }

// inline void AlgorithmConcrete::setErrorCallback(std::function<void(const std::string&)> callback) {
//     errorCallback_ = std::move(callback);
// }

// inline std::tuple<double, double> AlgorithmConcrete::getAlgorithmMetrics() const {
//     return {getLastFPS(), lastProcessingTime_};
// }

// inline double AlgorithmConcrete::getLastFPS() const {
//     return lastFPS_;
// }

// inline double AlgorithmConcrete::getFps() const {
//     return fps_;
// }

// inline double AlgorithmConcrete::getAverageProcTime() const {
//     return avgProcTime_;
// }

// inline const uint8_t* AlgorithmConcrete::getProcessedBuffer() const {
//     return processedBuffer_.data();
// }

// inline void AlgorithmConcrete::threadLoop() {
//     auto tStart = std::chrono::steady_clock::now();
//     while (running_) {
//         FrameData inputFrame;
//         FrameData outputFrame;
        
//         if (!inputQueue_->pop(inputFrame)) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(1));
//             continue;
//         }
        
//         if (processFrame(inputFrame, outputFrame)) {
//             outputQueue_->push(outputFrame);
//             framesCount_++;
//             double elapsedSec = std::chrono::duration<double>(
//                 std::chrono::steady_clock::now() - tStart).count();
//             if (elapsedSec >= 1.0) {
//                 updateMetrics(elapsedSec);
//                 tStart = std::chrono::steady_clock::now();
//                 framesCount_ = 0;
//                 totalProcTime_ = 0.0;
//             }
//         }
//     }
//     spdlog::info("[AlgorithmConcrete] Exiting FrameData thread loop ({}).", 
//                  algorithmTypeToString(algoConfig_.algorithmType));
// }


// inline void AlgorithmConcrete::threadLoopZeroCopy() {
//     auto tStart = std::chrono::steady_clock::now();
//     while (running_) {
//         std::shared_ptr<ZeroCopyFrameData> inputFrame;
//         std::shared_ptr<ZeroCopyFrameData> outputFrame;

//         if (!inputQueueZeroCopy_->pop(inputFrame)) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(1));
//             continue;
//         }

//         if (processFrameZeroCopy(inputFrame, outputFrame)) {
//             outputQueueZeroCopy_->push(outputFrame);
//             framesCount_++;
//             double elapsedSec = std::chrono::duration<double>(
//                 std::chrono::steady_clock::now() - tStart).count();
//             if (elapsedSec >= 1.0) {
//                 updateMetrics(elapsedSec);
//                 tStart = std::chrono::steady_clock::now();
//                 framesCount_ = 0;
//                 totalProcTime_ = 0.0;
//             }
//         }
//     }
//     spdlog::info("[AlgorithmConcrete] Exiting ZeroCopy thread loop ({}).", 
//                  algorithmTypeToString(algoConfig_.algorithmType));
// }

// inline bool AlgorithmConcrete::processFrame(const FrameData& inputFrame, FrameData& outputFrame) {
//     if (inputFrame.dataVec.empty() || inputFrame.size == 0) {
//         spdlog::warn("[AlgorithmConcrete] Invalid FrameData input.");
//         return false;
//     }

//     auto start = std::chrono::steady_clock::now();

//     outputFrame.frameNumber = inputFrame.frameNumber;
//     outputFrame.width = inputFrame.width;
//     outputFrame.height = inputFrame.height;
//     outputFrame.size = inputFrame.size;
//     outputFrame.dataVec.resize(inputFrame.size);

//     switch (algoConfig_.algorithmType) {
//         case AlgorithmType::Invert:
//             processInvert(inputFrame);
//             break;
//         case AlgorithmType::Grayscale:
//             processGrayscale(inputFrame);
//             break;
//         case AlgorithmType::EdgeDetection:
//             processEdgeDetection(inputFrame);
//             break;
//         case AlgorithmType::GaussianBlur:
//             processGaussianBlur(inputFrame);
//             break;
//         case AlgorithmType::MatrixMultiply:
//             processMatrixMultiply();
//             break;
//         case AlgorithmType::Mandelbrot:
//             processMandelbrot();
//             break;
//         case AlgorithmType::PasswordHash:
//             processPasswordHash(inputFrame);
//             break;
//         case AlgorithmType::MultiPipeline:
//             processMultiPipeline(inputFrame);
//             break;
//         case AlgorithmType::GPUMatrixMultiply:
//             processGPUMatrixMultiply();
//             break;
//         case AlgorithmType::MultiThreadedInvert:
//             processMultiThreadedInvert(inputFrame);
//             break;
//         default:
//             spdlog::warn("[AlgorithmConcrete] Unknown algorithm type.");
//             return false;
//     }

//     outputFrame.dataVec = processedBuffer_;
//     auto end = std::chrono::steady_clock::now();
//     lastProcessingTime_ = std::chrono::duration<double, std::milli>(end - start).count();
//     totalProcTime_ += lastProcessingTime_;
//     return true;
// }


// // AlgorithmConcrete.cpp
// //======================================================================================
// bool AlgorithmConcrete::processFrameZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& inputFrame,
//     std::shared_ptr<ZeroCopyFrameData>& outputFrame) {
// if (!inputFrame || !inputFrame->dataPtr) {
// spdlog::error("Invalid input frame in processFrameZeroCopy");
// return false;
// }

// // OutputFrame shares ownership with inputFrame (ZeroCopy)
// outputFrame = inputFrame; 

// // Process in-place
// switch (algoConfig_.algorithmType) {
// case AlgorithmType::Invert:
// processInvertZeroCopy(inputFrame);
// break;
// // ... other cases ...
// default:
// return false;
// }

// return true;
// }
// // //==============================Testing Code =================================================

// // // =====================================
// // inline bool AlgorithmConcrete::processFrameZeroCopy(
// //     const std::shared_ptr<ZeroCopyFrameData>& inputFrame,
// //     std::shared_ptr<ZeroCopyFrameData>& outputFrame)
// // {
// //     if (!inputFrame || !inputFrame->dataPtr || inputFrame->size == 0) {
// //         spdlog::warn("[AlgorithmConcrete] Invalid ZeroCopy input frame.");
// //         return false;
// //     }

// //     auto start = std::chrono::steady_clock::now();
    
// //     outputFrame = std::make_shared<ZeroCopyFrameData>();
// //     outputFrame->width = inputFrame->width;
// //     outputFrame->height = inputFrame->height;
// //     outputFrame->size = inputFrame->size;
// //     outputFrame->frameNumber = inputFrame->frameNumber;
// //     outputFrame->bufferIndex = inputFrame->bufferIndex;
// //     outputFrame->captureTime = inputFrame->captureTime;
    
// //     outputFrame->dataPtr = malloc(inputFrame->size);
// //     if (!outputFrame->dataPtr) {
// //         spdlog::error("[AlgorithmConcrete] Failed to allocate ZeroCopy output buffer.");
// //         return false;
// //     }

// //     switch (algoConfig_.algorithmType) {
// //         case AlgorithmType::Invert:
// //             processInvertZeroCopy(inputFrame);
// //             break;
// //         case AlgorithmType::Grayscale:
// //             processGrayscaleZeroCopy(inputFrame);
// //             break;
// //         case AlgorithmType::EdgeDetection:
// //             processEdgeDetectionZeroCopy(inputFrame);
// //             break;
// //         case AlgorithmType::GaussianBlur:
// //             processGaussianBlurZeroCopy(inputFrame);
// //             break;
// //         case AlgorithmType::MatrixMultiply:
// //             processMatrixMultiplyZeroCopy();
// //             break;
// //         case AlgorithmType::Mandelbrot:
// //             processMandelbrotZeroCopy();
// //             break;
// //         case AlgorithmType::PasswordHash:
// //             processPasswordHashZeroCopy(inputFrame);
// //             break;
// //         case AlgorithmType::MultiPipeline:
// //             processMultiPipelineZeroCopy(inputFrame);
// //             break;
// //         case AlgorithmType::GPUMatrixMultiply:
// //             processGPUMatrixMultiplyZeroCopy();
// //             break;
// //         case AlgorithmType::MultiThreadedInvert:
// //             processMultiThreadedInvertZeroCopy(inputFrame);
// //             break;
// //         default:
// //             spdlog::warn("[AlgorithmConcrete] Unknown algorithm type.");
// //             free(outputFrame->dataPtr);
// //             return false;
// //     }

// //     memcpy(outputFrame->dataPtr, processedBuffer_.data(), inputFrame->size);
// //     auto end = std::chrono::steady_clock::now();
// //     lastProcessingTime_ = std::chrono::duration<double, std::milli>(end - start).count();
// //     totalProcTime_ += lastProcessingTime_;
// //     return true;
// // }

// // //==============================Testing Code =================================================

// // inline bool AlgorithmConcrete::processFrameZeroCopy(
// //     const std::shared_ptr<ZeroCopyFrameData>& inputFrame,
// //     std::shared_ptr<ZeroCopyFrameData>& outputFrame)
// // {
// //     if (!inputFrame || !inputFrame->dataPtr || inputFrame->size == 0) {
// //         spdlog::warn("[AlgorithmConcrete] Invalid ZeroCopy input frame.");
// //         return false;
// //     }

// //     auto start = std::chrono::steady_clock::now();

// //     // Reuse the input frame's buffer for the output frame
// //     outputFrame = inputFrame;

// //     switch (algoConfig_.algorithmType) {
// //         case AlgorithmType::Invert:
// //             processInvertZeroCopy(inputFrame);
// //             break;
// //         case AlgorithmType::Grayscale:
// //             processGrayscaleZeroCopy(inputFrame);
// //             break;
// //         case AlgorithmType::EdgeDetection:
// //             processEdgeDetectionZeroCopy(inputFrame);
// //             break;
// //         case AlgorithmType::GaussianBlur:
// //             processGaussianBlurZeroCopy(inputFrame);
// //             break;
// //         default:
// //             spdlog::warn("[AlgorithmConcrete] Unknown algorithm type.");
// //             return false;
// //     }

// //     auto end = std::chrono::steady_clock::now();
// //     lastProcessingTime_ = std::chrono::duration<double, std::milli>(end - start).count();
// //     totalProcTime_ += lastProcessingTime_;

// //     return true;
// // }

// // //==============================Testing Code =================================================

// inline void AlgorithmConcrete::updateMetrics(double elapsedSec) {
//     fps_ = framesCount_ / elapsedSec;
//     avgProcTime_ = (framesCount_ > 0) ? (totalProcTime_ / framesCount_) : 0.0;
//     spdlog::info("[AlgorithmConcrete] FPS: {:.1f}, AvgTime: {:.2f} ms, Type: {}",
//                  fps_, avgProcTime_, algorithmTypeToString(algoConfig_.algorithmType));
//     PerformanceLogger::getInstance().pushAlgorithmStats(fps_, avgProcTime_);
//     lastFPS_ = fps_;
// }

// inline std::string AlgorithmConcrete::algorithmTypeToString(AlgorithmType type) const {
//     switch (type) {
//         case AlgorithmType::Invert: return "Invert";
//         case AlgorithmType::Grayscale: return "Grayscale";
//         case AlgorithmType::EdgeDetection: return "EdgeDetection";
//         case AlgorithmType::GaussianBlur: return "GaussianBlur";
//         case AlgorithmType::MatrixMultiply: return "MatrixMultiply";
//         case AlgorithmType::Mandelbrot: return "Mandelbrot";
//         case AlgorithmType::PasswordHash: return "PasswordHash";
//         case AlgorithmType::MultiPipeline: return "MultiPipeline";
//         case AlgorithmType::GPUMatrixMultiply: return "GPUMatrixMultiply";
//         case AlgorithmType::MultiThreadedInvert: return "MultiThreadedInvert";
//         default: return "Unknown";
//     }
// }

// inline void AlgorithmConcrete::reportError(const std::string& msg) {
//     if (errorCallback_) {
//         errorCallback_(msg);
//     } else {
//         spdlog::error("[AlgorithmConcrete] {}", msg);
//     }
// }

// inline void AlgorithmConcrete::parallelFor(size_t start, size_t end, std::function<void(size_t)> func) {
//     const size_t numThreads = algoConfig_.concurrencyLevel;
//     const size_t chunkSize = (end - start + numThreads - 1) / numThreads;
//     std::vector<std::thread> workers;

//     for (size_t i = 0; i < numThreads; ++i) {
//         size_t s = start + i * chunkSize;
//         size_t e = std::min(end, s + chunkSize);
//         if (s < e) {
//             workers.emplace_back([=] {
//                 for (size_t j = s; j < e; j++) {
//                     func(j);
//                 }
//             });
//         }
//     }
//     for (auto& t : workers) {
//         t.join();
//     }
// }

// // FrameData processing implementations
// inline void AlgorithmConcrete::processInvert(const FrameData& frame) {
//     processedBuffer_.resize(frame.size);
//     parallelFor(0, frame.size, [&](size_t i) {
//         processedBuffer_[i] = ~frame.dataVec[i];
//     });
// }

// inline void AlgorithmConcrete::processGrayscale(const FrameData& frame) {
//     processedBuffer_.resize(frame.size);
//     for (size_t i = 0; i < frame.size; i += 2) {
//         processedBuffer_[i] = frame.dataVec[i];
//         processedBuffer_[i + 1] = 128;
//     }
// }

// inline void AlgorithmConcrete::processEdgeDetection(const FrameData& frame) {
//     processedBuffer_.resize(frame.size);
//     int width = frame.width;
//     int height = frame.height;
//     for (int y = 0; y < height; ++y) {
//         for (int x = 0; x < width; x += 2) {
//             int idx = y * width * 2 + x * 2;
//             if (x > 0 && x < (width - 2)) {
//                 int grad = std::abs(frame.dataVec[idx] - frame.dataVec[idx - 2]);
//                 processedBuffer_[idx] = (uint8_t)grad;
//                 processedBuffer_[idx + 1] = 128;
//             } else {
//                 processedBuffer_[idx] = frame.dataVec[idx];
//                 processedBuffer_[idx + 1] = frame.dataVec[idx + 1];
//             }
//         }
//     }
// }

// inline void AlgorithmConcrete::processGaussianBlur(const FrameData& frame) {
//     const int radius = algoConfig_.blurRadius;
//     const int width = frame.width;
//     const int height = frame.height;
//     processedBuffer_.resize(frame.size);
//     std::vector<uint8_t> temp(frame.size);

//     parallelFor(0, height, [&](size_t y) {
//         for (int x = 0; x < width; x++) {
//             float sum = 0.f;
//             float wsum = 0.f;
//             for (int dx = -radius; dx <= radius; dx++) {
//                 int xx = std::clamp(x + dx, 0, width - 1);
//                 float w = std::exp(-(dx * dx) / (2.f * radius * radius));
//                 sum += frame.dataVec[y * width * 2 + xx * 2] * w;
//                 wsum += w;
//             }
//             temp[y * width * 2 + x * 2] = (uint8_t)(sum / wsum);
//             temp[y * width * 2 + x * 2 + 1] = 128;
//         }
//     });

//     parallelFor(0, width, [&](size_t x) {
//         for (int y = 0; y < height; y++) {
//             float sum = 0.f;
//             float wsum = 0.f;
//             for (int dy = -radius; dy <= radius; dy++) {
//                 int yy = std::clamp(y + dy, 0, height - 1);
//                 float w = std::exp(-(dy * dy) / (2.f * radius * radius));
//                 sum += temp[yy * width * 2 + x * 2] * w;
//                 wsum += w;
//             }
//             processedBuffer_[y * width * 2 + x * 2] = (uint8_t)(sum / wsum);
//             processedBuffer_[y * width * 2 + x * 2 + 1] = 128;
//         }
//     });
// }

// inline void AlgorithmConcrete::processMatrixMultiply() {
//     int N = algoConfig_.matrixSize;
//     std::vector<float> A(N * N, 1.0f);
//     std::vector<float> B(N * N, 2.0f);
//     std::vector<float> C(N * N, 0.0f);

//     parallelFor(0, N, [&](size_t i) {
//         for (int k = 0; k < N; k++) {
//             for (int j = 0; j < N; j++) {
//                 C[i * N + j] += A[i * N + k] * B[k * N + j];
//             }
//         }
//     });

//     processedBuffer_.resize(N * N);
//     for (int i = 0; i < N * N; i++) {
//         processedBuffer_[i] = (uint8_t)std::min(255.f, C[i] * 0.01f);
//     }
// }

// inline void AlgorithmConcrete::processMandelbrot() {
//     const int width = 256;
//     const int height = 256;
//     processedBuffer_.resize(width * height * 2, 0);

//     parallelFor(0, height, [&](size_t py) {
//         float y0 = (py - height / 2.0f) * 4.0f / width;
//         for (int px = 0; px < width; px++) {
//             float x0 = (px - width / 2.0f) * 4.0f / width;
//             float x = 0.0f;
//             float y = 0.0f;
//             int iteration = 0;
//             int maxIter = algoConfig_.mandelbrotIter;

//             while (x * x + y * y < 4 && iteration < maxIter) {
//                 float xtemp = x * x - y * y + x0;
//                 y = 2 * x * y + y0;
//                 x = xtemp;
//                 iteration++;
//             }
//             processedBuffer_[py * width * 2 + px * 2] = (uint8_t)((iteration * 255) / maxIter);
//             processedBuffer_[py * width * 2 + px * 2 + 1] = 128;
//         }
//     });
// }

// inline void AlgorithmConcrete::processPasswordHash(const FrameData& frame) {
//     processedBuffer_.resize(frame.size);
//     int iterations = 10000;
//     parallelFor(0, frame.size, [&](size_t i) {
//         uint32_t hash = (uint32_t)frame.dataVec[i];
//         for (int j = 0; j < iterations; j++) {
//             hash = (hash << 5) + hash + j;
//         }
//         processedBuffer_[i] = (uint8_t)(hash % 256);
//     });
// }

// inline void AlgorithmConcrete::processMultiPipeline(const FrameData& frame) {
//     processedBuffer_.resize(frame.size);
//     for (size_t i = 0; i < frame.size; i++) {
//         processedBuffer_[i] = ~frame.dataVec[i];
//     }
//     for (size_t i = 0; i < frame.size; i += 2) {
//         processedBuffer_[i + 1] = 128;
//     }
// }

// inline void AlgorithmConcrete::processGPUMatrixMultiply() {
//     spdlog::warn("[AlgorithmConcrete] GPUMatrixMultiply not implemented.");
//     processedBuffer_.resize(1024 * 2, 128);
// }

// inline void AlgorithmConcrete::processMultiThreadedInvert(const FrameData& frame) {
//     processedBuffer_.resize(frame.size);
//     const size_t chunkSize = frame.size / algoConfig_.concurrencyLevel;
//     std::vector<std::future<void>> futures;
//     for (int i = 0; i < algoConfig_.concurrencyLevel; i++) {
//         size_t start = i * chunkSize;
//         size_t end = (i == algoConfig_.concurrencyLevel - 1) ? frame.size : (start + chunkSize);
//         futures.push_back(std::async(std::launch::async, [this, &frame, start, end] {
//             for (size_t j = start; j < end; j++) {
//                 processedBuffer_[j] = ~frame.dataVec[j];
//             }
//         }));
//     }
//     for (auto& f : futures) {
//         f.get();
//     }
// }

// //ZeroCopyFrameData processing implementations

// // inline void AlgorithmConcrete::processInvertZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
// //     processedBuffer_.resize(frame->size);
// //     uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
// //     parallelFor(0, frame->size, [&](size_t i) { processedBuffer_[i] = ~data[i];});
// // }

// inline void AlgorithmConcrete::processInvertZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
//     if (!frame || !frame->dataPtr) {
//         spdlog::error("[AlgorithmConcrete] Invalid frame buffer in processInvertZeroCopy");
//         return;
//     }

//     uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
//     parallelFor(0, frame->size, [&](size_t i) { data[i] = ~data[i]; });
// }

// //=========================================================================================================
// //---
// //````
// //```

// // // Step 1: Validate dataPtr Before Processing
// // inline void AlgorithmConcrete::processInvertZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
// //     if (!frame || !frame->dataPtr) {
// //         spdlog::error("[AlgorithmConcrete] Invalid ZeroCopyFrameData: dataPtr is null.");
// //         return;
// //     }

// //     uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
// //     parallelFor(0, frame->size, [&](size_t i) { data[i] = ~data[i]; });
// // }


// //==========================================================================================================
// inline void AlgorithmConcrete::processGrayscaleZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
//     processedBuffer_.resize(frame->size);
//     uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
//     for (size_t i = 0; i < frame->size; i += 2) {
//         processedBuffer_[i] = data[i];
//         processedBuffer_[i + 1] = 128;
//     }
// }

// inline void AlgorithmConcrete::processEdgeDetectionZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
//     processedBuffer_.resize(frame->size);
//     uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
//     int width = frame->width;
//     int height = frame->height;
//     for (int y = 0; y < height; ++y) {
//         for (int x = 0; x < width; x += 2) {
//             int idx = y * width * 2 + x * 2;
//             if (x > 0 && x < (width - 2)) {
//                 int grad = std::abs(data[idx] - data[idx - 2]);
//                 processedBuffer_[idx] = (uint8_t)grad;
//                 processedBuffer_[idx + 1] = 128;
//             } else {
//                 processedBuffer_[idx] = data[idx];
//                 processedBuffer_[idx + 1] = data[idx + 1];
//             }
//         }
//     }
// }

// inline void AlgorithmConcrete::processGaussianBlurZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
//     const int radius = algoConfig_.blurRadius;
//     const int width = frame->width;
//     const int height = frame->height;
//     processedBuffer_.resize(frame->size);
//     std::vector<uint8_t> temp(frame->size);
//     uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);

//     parallelFor(0, height, [&](size_t y) {
//         for (int x = 0; x < width; x++) {
//             float sum = 0.f;
//             float wsum = 0.f;
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

//     parallelFor(0, width, [&](size_t x) {
//         for (int y = 0; y < height; y++) {
//             float sum = 0.f;
//             float wsum = 0.f;
//             for (int dy = -radius; dy <= radius; dy++) {
//                 int yy = std::clamp(y + dy, 0, height - 1);
//                 float w = std::exp(-(dy * dy) / (2.f * radius * radius));
//                 sum += temp[yy * width * 2 + x * 2] * w;
//                 wsum += w;
//             }
//             processedBuffer_[y * width * 2 + x * 2] = (uint8_t)(sum / wsum);
//             processedBuffer_[y * width * 2 + x * 2 + 1] = 128;
//         }
//     });
// }

// inline void AlgorithmConcrete::processMatrixMultiplyZeroCopy() {
//     int N = algoConfig_.matrixSize;
//     std::vector<float> A(N * N, 1.0f);
//     std::vector<float> B(N * N, 2.0f);
//     std::vector<float> C(N * N, 0.0f);

//     parallelFor(0, N, [&](size_t i) {
//         for (int k = 0; k < N; k++) {
//             for (int j = 0; j < N; j++) {
//                 C[i * N + j] += A[i * N + k] * B[k * N + j];
//             }
//         }
//     });

//     processedBuffer_.resize(N * N);
//     for (int i = 0; i < N * N; i++) {
//         processedBuffer_[i] = (uint8_t)std::min(255.f, C[i] * 0.01f);
//     }
// }

// inline void AlgorithmConcrete::processMandelbrotZeroCopy() {
//     const int width = 256;
//     const int height = 256;
//     processedBuffer_.resize(width * height * 2, 0);

//     parallelFor(0, height, [&](size_t py) {
//         float y0 = (py - height / 2.0f) * 4.0f / width;
//         for (int px = 0; px < width; px++) {
//             float x0 = (px - width / 2.0f) * 4.0f / width;
//             float x = 0.0f;
//             float y = 0.0f;
//             int iteration = 0;
//             int maxIter = algoConfig_.mandelbrotIter;

//             while (x * x + y * y < 4 && iteration < maxIter) {
//                 float xtemp = x * x - y * y + x0;
//                 y = 2 * x * y + y0;
//                 x = xtemp;
//                 iteration++;
//             }
//             processedBuffer_[py * width * 2 + px * 2] = (uint8_t)((iteration * 255) / maxIter);
//             processedBuffer_[py * width * 2 + px * 2 + 1] = 128;
//         }
//     });
// }

// inline void AlgorithmConcrete::processPasswordHashZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
//     processedBuffer_.resize(frame->size);
//     uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
//     int iterations = 10000;
//     parallelFor(0, frame->size, [&](size_t i) {
//         uint32_t hash = (uint32_t)data[i];
//         for (int j = 0; j < iterations; j++) {
//             hash = (hash << 5) + hash + j;
//         }
//         processedBuffer_[i] = (uint8_t)(hash % 256);
//     });
// }

// inline void AlgorithmConcrete::processMultiPipelineZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
//     processedBuffer_.resize(frame->size);
//     uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
//     for (size_t i = 0; i < frame->size; i++) {
//         processedBuffer_[i] = ~data[i];
//     }
//     for (size_t i = 0; i < frame->size; i += 2) {
//         processedBuffer_[i + 1] = 128;
//     }
// }

// inline void AlgorithmConcrete::processGPUMatrixMultiplyZeroCopy() {
//     spdlog::warn("[AlgorithmConcrete] GPUMatrixMultiplyZeroCopy not implemented.");
//     processedBuffer_.resize(1024 * 2, 128);
// }

// inline void AlgorithmConcrete::processMultiThreadedInvertZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
//     processedBuffer_.resize(frame->size);
//     uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
//     const size_t chunkSize = frame->size / algoConfig_.concurrencyLevel;
//     std::vector<std::future<void>> futures;
//     for (int i = 0; i < algoConfig_.concurrencyLevel; i++) {
//         size_t start = i * chunkSize;
//         size_t end = (i == algoConfig_.concurrencyLevel - 1) ? frame->size : (start + chunkSize);
//         futures.push_back(std::async(std::launch::async, [this, data, start, end] {
//             for (size_t j = start; j < end; j++) {
//                 processedBuffer_[j] = ~data[j];
//             }
//         }));
//     }
//     for (auto& f : futures) {
//         f.get();
//     }
// }


//=================== Woring final version

// #pragma once

// #include "../Interfaces/IAlgorithm.h"
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/ZeroCopyFrameData.h"
// #include "../SharedStructures/SharedQueue.h"
// #include "../SharedStructures/ThreadManager.h"
// #include "../../Stage_02/Logger/PerformanceLogger.h"
// #include "../SharedStructures/AlgorithmConfig.h"

// #include <vector>
// #include <mutex>
// #include <atomic>
// #include <functional>
// #include <spdlog/spdlog.h>
// #include <chrono>
// #include <future>
// #include <cmath>
// #include <thread>

// class AlgorithmConcrete : public IAlgorithm {
// public:
//     explicit AlgorithmConcrete(ThreadManager& threadManager);
//     AlgorithmConcrete(std::shared_ptr<SharedQueue<FrameData>> inputQueue,
//                      std::shared_ptr<SharedQueue<FrameData>> outputQueue,
//                      ThreadManager& threadManager);
//     AlgorithmConcrete(std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
//                      std::vector<std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>> outputQueues,
//                      ThreadManager& threadManager);

//     ~AlgorithmConcrete() override;

//     static std::shared_ptr<IAlgorithm> createAlgorithm(
//         AlgorithmType type,
//         std::shared_ptr<SharedQueue<FrameData>> inputQueue,
//         std::shared_ptr<SharedQueue<FrameData>> outputQueue,
//         ThreadManager& threadManager);
//     static std::shared_ptr<IAlgorithm> createAlgorithmZeroCopy(
//         AlgorithmType type,
//         std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
//         std::vector<std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>> outputQueues,
//         ThreadManager& threadManager);

//     void startAlgorithm() override;
//     void stopAlgorithm() override;
//     bool processFrame(const FrameData& inputFrame, FrameData& outputFrame) override;
//     bool processFrameZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& inputFrame,
//                             std::vector<std::shared_ptr<ZeroCopyFrameData>>& outputFrames) override;
//     bool configure(const AlgorithmConfig& config) override;
//     void setErrorCallback(std::function<void(const std::string&)> callback) override;
//     std::tuple<double, double> getAlgorithmMetrics() const override;
//     double getLastFPS() const override;
//     double getFps() const override;
//     double getAverageProcTime() const override;
//     const uint8_t* getProcessedBuffer() const override;

// private:
//     void threadLoop();
//     void threadLoopZeroCopy();
//     void updateMetrics(double elapsedSec);
//     std::string algorithmTypeToString(AlgorithmType type) const;
//     void reportError(const std::string& msg);

//     // Filter functions for ZeroCopyFrameData
//     void filterGrayscale(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height);
//     void filterInvert(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height);
//     void filterSepia(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height);
//     void filterEdge(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height);

//     std::atomic<bool> running_{false};
//     std::thread algoThread_;

//     std::shared_ptr<SharedQueue<FrameData>> inputQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> outputQueue_;
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueueZeroCopy_;
//     std::vector<std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>> outputQueuesZeroCopy_;

//     AlgorithmConfig algoConfig_;
//     std::function<void(const std::string&)> errorCallback_;

//     mutable std::mutex metricMutex_;
//     double fps_ = 0.0;
//     double avgProcTime_ = 0.0;
//     int framesCount_ = 0;
//     double totalProcTime_ = 0.0;

//     std::vector<uint8_t> processedBuffer_;

//     mutable std::mutex fpsMutex_;
//     mutable std::atomic<int> framesProcessed_{0};
//     mutable std::chrono::steady_clock::time_point lastUpdateTime_;
//     mutable double lastFPS_ = 0.0;
//     mutable double lastProcessingTime_ = 0.0;

//     ThreadManager& threadManager_;
// };

// // Implementation
// inline AlgorithmConcrete::AlgorithmConcrete(ThreadManager& threadManager)
//     : running_{false}
//     , inputQueue_(nullptr)
//     , outputQueue_(nullptr)
//     , inputQueueZeroCopy_(nullptr)
//     , lastUpdateTime_(std::chrono::steady_clock::now())
//     , threadManager_(threadManager)
// {
//     spdlog::warn("[AlgorithmConcrete] Default constructor called - no queues provided.");
// }

// inline AlgorithmConcrete::AlgorithmConcrete(std::shared_ptr<SharedQueue<FrameData>> inputQueue,
//                                             std::shared_ptr<SharedQueue<FrameData>> outputQueue,
//                                             ThreadManager& threadManager)
//     : running_{false}
//     , inputQueue_(std::move(inputQueue))
//     , outputQueue_(std::move(outputQueue))
//     , inputQueueZeroCopy_(nullptr)
//     , lastUpdateTime_(std::chrono::steady_clock::now())
//     , threadManager_(threadManager)
// {
//     spdlog::info("[AlgorithmConcrete] FrameData constructor called.");
// }

// inline AlgorithmConcrete::AlgorithmConcrete(
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
//     std::vector<std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>> outputQueues,
//     ThreadManager& threadManager)
//     : running_{false}
//     , inputQueue_(nullptr)
//     , outputQueue_(nullptr)
//     , inputQueueZeroCopy_(std::move(inputQueue))
//     , outputQueuesZeroCopy_(std::move(outputQueues))
//     , lastUpdateTime_(std::chrono::steady_clock::now())
//     , threadManager_(threadManager)
// {
//     spdlog::info("[AlgorithmConcrete] ZeroCopy constructor called with multiple output queues.");
// }

// inline AlgorithmConcrete::~AlgorithmConcrete() {
//     stopAlgorithm();
// }

// inline std::shared_ptr<IAlgorithm> AlgorithmConcrete::createAlgorithm(
//     AlgorithmType type,
//     std::shared_ptr<SharedQueue<FrameData>> inputQueue,
//     std::shared_ptr<SharedQueue<FrameData>> outputQueue,
//     ThreadManager& threadManager)
// {
//     if (!inputQueue || !outputQueue) {
//         spdlog::error("[AlgorithmConcrete] FrameData queues must be non-null.");
//         return nullptr;
//     }
//     auto algo = std::make_shared<AlgorithmConcrete>(inputQueue, outputQueue, threadManager);
//     AlgorithmConfig cfg;
//     cfg.algorithmType = type;
//     algo->configure(cfg);
//     return algo;
// }

// inline std::shared_ptr<IAlgorithm> AlgorithmConcrete::createAlgorithmZeroCopy(
//     AlgorithmType type,
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
//     std::vector<std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>> outputQueues,
//     ThreadManager& threadManager)
// {
//     if (!inputQueue || outputQueues.empty()) {
//         spdlog::error("[AlgorithmConcrete] ZeroCopy queues must be non-null and non-empty.");
//         return nullptr;
//     }
//     auto algo = std::make_shared<AlgorithmConcrete>(inputQueue, outputQueues, threadManager);
//     AlgorithmConfig cfg;
//     cfg.algorithmType = type;
//     algo->configure(cfg);
//     return algo;
// }

// inline void AlgorithmConcrete::startAlgorithm() {
//     if (running_) {
//         spdlog::warn("[AlgorithmConcrete] Algorithm is already running.");
//         return;
//     }
    
//     running_ = true;
//     if (inputQueueZeroCopy_ && !outputQueuesZeroCopy_.empty()) {
//         threadManager_.addThread("AlgorithmProcessingZeroCopy",
//             std::thread(&AlgorithmConcrete::threadLoopZeroCopy, this));
//     } else if (inputQueue_ && outputQueue_) {
//         threadManager_.addThread("AlgorithmProcessing",
//             std::thread(&AlgorithmConcrete::threadLoop, this));
//     } else {
//         spdlog::error("[AlgorithmConcrete] No valid queues initialized.");
//         running_ = false;
//         return;
//     }
    
//     spdlog::info("[AlgorithmConcrete] Algorithm thread started ({}).",
//                  algorithmTypeToString(algoConfig_.algorithmType));
// }

// inline void AlgorithmConcrete::stopAlgorithm() {
//     if (!running_) return;
//     running_ = false;
//     if (outputQueue_) {
//         outputQueue_->stop();
//     }
//     for (auto& queue : outputQueuesZeroCopy_) {
//         if (queue) queue->stop();
//     }
//     threadManager_.joinThreadsFor("AlgorithmProcessing");
//     threadManager_.joinThreadsFor("AlgorithmProcessingZeroCopy");
//     spdlog::info("[AlgorithmConcrete] Algorithm thread stopped.");
// }

// inline bool AlgorithmConcrete::configure(const AlgorithmConfig& config) {
//     algoConfig_ = config;
//     spdlog::info("[AlgorithmConcrete] Configured algorithm: {}", 
//                  algorithmTypeToString(config.algorithmType));
//     return true;
// }

// inline void AlgorithmConcrete::setErrorCallback(std::function<void(const std::string&)> callback) {
//     errorCallback_ = std::move(callback);
// }

// inline std::tuple<double, double> AlgorithmConcrete::getAlgorithmMetrics() const {
//     return {getLastFPS(), lastProcessingTime_};
// }

// inline double AlgorithmConcrete::getLastFPS() const {
//     return lastFPS_;
// }

// inline double AlgorithmConcrete::getFps() const {
//     return fps_;
// }

// inline double AlgorithmConcrete::getAverageProcTime() const {
//     return avgProcTime_;
// }

// inline const uint8_t* AlgorithmConcrete::getProcessedBuffer() const {
//     return processedBuffer_.data();
// }

// inline void AlgorithmConcrete::threadLoop() {
//     auto tStart = std::chrono::steady_clock::now();
//     while (running_) {
//         FrameData inputFrame;
//         FrameData outputFrame;
        
//         if (!inputQueue_->pop(inputFrame)) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(1));
//             continue;
//         }
        
//         if (processFrame(inputFrame, outputFrame)) {
//             outputQueue_->push(outputFrame);
//             framesCount_++;
//             double elapsedSec = std::chrono::duration<double>(
//                 std::chrono::steady_clock::now() - tStart).count();
//             if (elapsedSec >= 1.0) {
//                 updateMetrics(elapsedSec);
//                 tStart = std::chrono::steady_clock::now();
//                 framesCount_ = 0;
//                 totalProcTime_ = 0.0;
//             }
//         }
//     }
//     spdlog::info("[AlgorithmConcrete] Exiting FrameData thread loop ({}).", 
//                  algorithmTypeToString(algoConfig_.algorithmType));
// }

// inline void AlgorithmConcrete::threadLoopZeroCopy() {
//     auto tStart = std::chrono::steady_clock::now();
//     while (running_) {
//         std::shared_ptr<ZeroCopyFrameData> inputFrame;
//         std::vector<std::shared_ptr<ZeroCopyFrameData>> outputFrames;

//         if (!inputQueueZeroCopy_->pop(inputFrame)) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(1));
//             continue;
//         }

//         if (processFrameZeroCopy(inputFrame, outputFrames)) {
//             if (outputFrames.size() == outputQueuesZeroCopy_.size()) {
//                 for (size_t i = 0; i < outputFrames.size(); ++i) {
//                     outputQueuesZeroCopy_[i]->push(outputFrames[i]);
//                 }
//             }
//             framesCount_++;
//             double elapsedSec = std::chrono::duration<double>(
//                 std::chrono::steady_clock::now() - tStart).count();
//             if (elapsedSec >= 1.0) {
//                 updateMetrics(elapsedSec);
//                 tStart = std::chrono::steady_clock::now();
//                 framesCount_ = 0;
//                 totalProcTime_ = 0.0;
//             }
//         }
//     }
//     spdlog::info("[AlgorithmConcrete] Exiting ZeroCopy thread loop ({}).", 
//                  algorithmTypeToString(algoConfig_.algorithmType));
// }

// inline bool AlgorithmConcrete::processFrame(const FrameData& inputFrame, FrameData& outputFrame) {
//     // Placeholder for FrameData processing
//     return false;
// }

// inline bool AlgorithmConcrete::processFrameZeroCopy(
//     const std::shared_ptr<ZeroCopyFrameData>& inputFrame,
//     std::vector<std::shared_ptr<ZeroCopyFrameData>>& outputFrames)
// {
//     if (!inputFrame || !inputFrame->dataPtr) return false;

//     // The input frame is in YUYV format; SdlDisplayConcrete will convert it to RGB
//     // For processing, we assume SdlDisplayConcrete has already converted the original frame to RGB
//     // Here, we just pass the YUYV data through and let the display handle conversion
//     // However, for filtering, we need RGB data, so we'll assume the display provides RGB data
//     // In this context, we'll process the frame as if it's already RGB (post-conversion by SdlDisplayConcrete)

//     std::vector<uint8_t> rgbData(static_cast<uint8_t*>(inputFrame->dataPtr),
//                                  static_cast<uint8_t*>(inputFrame->dataPtr) + inputFrame->size);

//     // Create buffers for each filter
//     std::vector<uint8_t> filteredGray(inputFrame->size);
//     std::vector<uint8_t> filteredInvert(inputFrame->size);
//     std::vector<uint8_t> filteredSepia(inputFrame->size);
//     std::vector<uint8_t> filteredEdge(inputFrame->size);

//     // Apply filters in parallel
//     std::vector<std::thread> filterThreads;
//     filterThreads.emplace_back([this, &rgbData, &filteredGray, w = inputFrame->width, h = inputFrame->height]() {
//         filterGrayscale(rgbData, filteredGray, w, h);
//     });
//     filterThreads.emplace_back([this, &rgbData, &filteredInvert, w = inputFrame->width, h = inputFrame->height]() {
//         filterInvert(rgbData, filteredInvert, w, h);
//     });
//     filterThreads.emplace_back([this, &rgbData, &filteredSepia, w = inputFrame->width, h = inputFrame->height]() {
//         filterSepia(rgbData, filteredSepia, w, h);
//     });
//     filterThreads.emplace_back([this, &rgbData, &filteredEdge, w = inputFrame->width, h = inputFrame->height]() {
//         filterEdge(rgbData, filteredEdge, w, h);
//     });

//     for (auto& thread : filterThreads) {
//         if (thread.joinable()) thread.join();
//     }

//     // Create output frames
//     outputFrames.clear();
//     std::vector<std::vector<uint8_t>> filteredData = {filteredGray, filteredInvert, filteredSepia, filteredEdge};
//     for (const auto& filtered : filteredData) {
//         auto outputFrame = std::make_shared<ZeroCopyFrameData>(*inputFrame);
//         outputFrame->dataPtr = new uint8_t[filtered.size()];
//         outputFrame->size = filtered.size();
//         std::copy(filtered.begin(), filtered.end(), static_cast<uint8_t*>(outputFrame->dataPtr));
//         outputFrames.push_back(outputFrame);
//     }

//     return true;
// }

// inline void AlgorithmConcrete::filterGrayscale(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height) {
//     dst.resize(width * height * 3);
//     for (size_t i = 0; i < src.size(); i += 3) {
//         uint8_t r = src[i];
//         uint8_t g = src[i+1];
//         uint8_t b = src[i+2];
//         uint8_t gray = static_cast<uint8_t>(0.299*r + 0.587*g + 0.114*b);
//         dst[i] = dst[i+1] = dst[i+2] = gray;
//     }
// }

// inline void AlgorithmConcrete::filterInvert(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height) {
//     dst.resize(width * height * 3);
//     for (size_t i = 0; i < src.size(); ++i) {
//         dst[i] = 255 - src[i];
//     }
// }

// inline void AlgorithmConcrete::filterSepia(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height) {
//     dst.resize(width * height * 3);
//     for (size_t i = 0; i < src.size(); i += 3) {
//         uint8_t r = src[i];
//         uint8_t g = src[i+1];
//         uint8_t b = src[i+2];
//         int tr = static_cast<int>(0.393*r + 0.769*g + 0.189*b);
//         int tg = static_cast<int>(0.349*r + 0.686*g + 0.168*b);
//         int tb = static_cast<int>(0.272*r + 0.534*g + 0.131*b);
//         dst[i]   = static_cast<uint8_t>(std::min(255, tr));
//         dst[i+1] = static_cast<uint8_t>(std::min(255, tg));
//         dst[i+2] = static_cast<uint8_t>(std::min(255, tb));
//     }
// }

// inline void AlgorithmConcrete::filterEdge(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int width, int height) {
//     std::vector<uint8_t> gray(width * height);
//     for (int i = 0; i < width * height; i++) {
//         int idx = i * 3;
//         uint8_t r = src[idx];
//         uint8_t g = src[idx+1];
//         uint8_t b = src[idx+2];
//         gray[i] = static_cast<uint8_t>(0.299*r + 0.587*g + 0.114*b);
//     }
//     dst.resize(width * height * 3);
//     for (int y = 0; y < height; y++) {
//         for (int x = 0; x < width; x++) {
//             int idx = y * width + x;
//             int diff = (x < width - 1) ? std::abs(gray[idx] - gray[idx+1]) : 0;
//             uint8_t edge = (diff > 20) ? 255 : 0;
//             int rgbIdx = idx * 3;
//             dst[rgbIdx] = dst[rgbIdx+1] = dst[rgbIdx+2] = edge;
//         }
//     }
// }

// inline void AlgorithmConcrete::updateMetrics(double elapsedSec) {
//     fps_ = framesCount_ / elapsedSec;
//     avgProcTime_ = (framesCount_ > 0) ? (totalProcTime_ / framesCount_) : 0.0;
//     spdlog::info("[AlgorithmConcrete] FPS: {:.1f}, AvgTime: {:.2f} ms, Type: {}",
//                  fps_, avgProcTime_, algorithmTypeToString(algoConfig_.algorithmType));
//     PerformanceLogger::getInstance().pushAlgorithmStats(fps_, avgProcTime_);
//     lastFPS_ = fps_;
// }

// inline std::string AlgorithmConcrete::algorithmTypeToString(AlgorithmType type) const {
//     switch (type) {
//         case AlgorithmType::Invert: return "Invert";
//         case AlgorithmType::Grayscale: return "Grayscale";
//         case AlgorithmType::EdgeDetection: return "EdgeDetection";
//         case AlgorithmType::GaussianBlur: return "GaussianBlur";
//         case AlgorithmType::MatrixMultiply: return "MatrixMultiply";
//         case AlgorithmType::Mandelbrot: return "Mandelbrot";
//         case AlgorithmType::PasswordHash: return "PasswordHash";
//         case AlgorithmType::MultiPipeline: return "MultiPipeline";
//         case AlgorithmType::GPUMatrixMultiply: return "GPUMatrixMultiply";
//         case AlgorithmType::MultiThreadedInvert: return "MultiThreadedInvert";
//         default: return "Unknown";
//     }
// }

// inline void AlgorithmConcrete::reportError(const std::string& msg) {
//     if (errorCallback_) {
//         errorCallback_(msg);
//     } else {
//         spdlog::error("[AlgorithmConcrete] {}", msg);
//     }
// }

//============================================================================================

///=====================TESTING  code Final Version ==============================
// // AlgorithmConcrete_new.h

#pragma once

#include "../Interfaces/IAlgorithm.h"
#include "../SharedStructures/FrameData.h"
#include "../SharedStructures/ZeroCopyFrameData.h"
#include "../SharedStructures/SharedQueue.h"
#include "../SharedStructures/ThreadManager.h"
#include "../SharedStructures/AlgorithmConfig.h"
//#include "../../Stage_02/Logger/PerformanceLogger.h"

#include "AlgorithmConcreteKernels.cuh" // Include CUDA header

#include "../SharedStructures/LucasKanadeOpticalFlow.h"
#include "../SharedStructures/allModulesStatcs.h"

#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <functional>
#include <spdlog/spdlog.h>
#include <chrono>
#include <future>
#include <cmath>
#include <algorithm>
#include <memory>



// Cuda Library

#include "../../usr/local/cuda-10.2/include/cuda_runtime.h"
//#include "../../usr/local/cuda-10.2/include/cuda_runtime_api.h"        
#include "../../usr/local/cuda-10.2/include/device_launch_parameters.h"

/**
 * @class AlgorithmConcrete
 * @brief Implements IAlgorithm for both copy-based (FrameData) and zero-copy frames (ZeroCopyFrameData).
 *
 * Depending on which constructor is used, you either set up:
 * - Copy-based queues (FrameData)
 * - Zero-copy queues (std::shared_ptr<ZeroCopyFrameData>)
 *
 * An internal thread is spawned (via ThreadManager) to pop frames from the input queue,
 * process them, and push them onto the output queue. Various algorithm modes (Invert,
 * Grayscale, etc.) are selected by configuring with an AlgorithmConfig.
 */
class AlgorithmConcrete : public IAlgorithm {
public:
    /**
     * @brief Default constructor (no queues). Not typically used, logs a warning.
     */
    explicit AlgorithmConcrete(ThreadManager& threadManager);

    /**
     * @brief Constructor for copy-based frames
     * @param inputQueue   SharedQueue<FrameData> for input
     * @param outputQueue  SharedQueue<FrameData> for output
     * @param threadManager Reference to ThreadManager
     */
    AlgorithmConcrete(std::shared_ptr<SharedQueue<FrameData>> inputQueue,
                      std::shared_ptr<SharedQueue<FrameData>> outputQueue,
                      ThreadManager& threadManager);

    /**
     * @brief Constructor for zero-copy frames
     * @param inputQueue   SharedQueue<std::shared_ptr<ZeroCopyFrameData>> for input
     * @param outputQueue  SharedQueue<std::shared_ptr<ZeroCopyFrameData>> for output
     * @param threadManager Reference to ThreadManager
     */
    AlgorithmConcrete(std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
                      std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> outputQueue,
                      ThreadManager& threadManager,
                      std::shared_ptr<ISystemMetricsAggregator> aggregator
                    );

    /**
     * @brief Destructor (stops algorithm if running)
     */
    ~AlgorithmConcrete() override;

    // ---------------------- Factory Methods ----------------------
    static std::shared_ptr<IAlgorithm> createAlgorithm(
        AlgorithmType type,
        std::shared_ptr<SharedQueue<FrameData>> inputQueue,
        std::shared_ptr<SharedQueue<FrameData>> outputQueue,
        ThreadManager& threadManager);

    static std::shared_ptr<IAlgorithm> createAlgorithmZeroCopy(
        AlgorithmType type,
        std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
        std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> outputQueue,
        ThreadManager& threadManager,
        std::shared_ptr<ISystemMetricsAggregator> aggregator);

    // ---------------------- IAlgorithm Interface ----------------------
    void startAlgorithm() override;
    void stopAlgorithm() override;

    bool processFrame(const FrameData& inputFrame, FrameData& outputFrame) override;
    bool processFrameZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& inputFrame,
                              std::shared_ptr<ZeroCopyFrameData>& outputFrame) override;

    bool configure(const AlgorithmConfig& config) override;
    void setErrorCallback(std::function<void(const std::string&)>) override;

    std::tuple<double, double> getAlgorithmMetrics() const override; // (FPS, lastProcTime)
    double getLastFPS() const override;         // last computed FPS
    double getFps() const override;             // an alternative or same as above
    double getAverageProcTime() const override; // average processing time
    const uint8_t* getProcessedBuffer() const override;


    // ---------------------- Dynamic Runtime Algorithm Switching ----------------------
    void setAlgorithmType(AlgorithmType newType);

private:


    

    // ---------------------- Thread Loops ----------------------
    void threadLoop();          ///< Processes FrameData
    void threadLoopZeroCopy();  ///< Processes ZeroCopyFrameData

    // ---------------------- Helpers ----------------------
    void updateMetrics(double elapsedSec);
    std::string algorithmTypeToString(AlgorithmType type) const;
    void reportError(const std::string& msg);

    // A simple chunk-based parallel for
    void parallelFor(size_t start, size_t end, std::function<void(size_t)> func);

    // ---------------------- Processing (Copy-based) ----------------------
     // CPU-based processing (FrameData)
    void processInvert(const FrameData& frame);
    void processGrayscale(const FrameData& frame);
    void processEdgeDetection(const FrameData& frame);
    void processGaussianBlur(const FrameData& frame);
    void processMatrixMultiply();
    void processMandelbrot();
    void processPasswordHash(const FrameData& frame);
    void processMultiPipeline(const FrameData& frame);
    void processGPUMatrixMultiply();
    void processMultiThreadedInvert(const FrameData& frame);

    // ---------------------- Processing (ZeroCopy) ----------------------
    void processInvertZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
    void processGrayscaleZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
    void processEdgeDetectionZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
    void processGaussianBlurZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
    void processMatrixMultiplyZeroCopy();
    void processMandelbrotZeroCopy();
    void processPasswordHashZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
    void processMultiPipelineZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
    void processGPUMatrixMultiplyZeroCopy();
    void processMultiThreadedInvertZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);

    // ---------------------- Processing (Lucas-Kanade Optical Flow) ----------------------
    std::unique_ptr<LucasKanadeOpticalFlow> opticalFlowProcessor_;
    std::shared_ptr<ZeroCopyFrameData> previousFrame_;
    void processOpticalFlow(const std::shared_ptr<ZeroCopyFrameData>& frame);

    // ---------------------- Processing (CUDA) ----------------------
    // New GPU-based filters
    // void processSobelEdgeZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
    // void processMedianFilterZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);
    // void processHistogramEqualizationZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);

    // Heterogeneous CPU+GPU filter
    // void processHeterogeneousGaussianBlurZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame);

    // CUDA processing methods updated to use wrapper functions
    void processSobelEdgeZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
        AlgorithmConcreteKernels::launchSobelEdgeKernel(frame, processedBuffer_);
    }

    void processMedianFilterZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
        AlgorithmConcreteKernels::launchMedianFilterKernel(frame, processedBuffer_, algoConfig_.medianWindowSize);
    }

    void processHistogramEqualizationZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
        AlgorithmConcreteKernels::launchHistogramEqualizationKernel(frame, processedBuffer_);
    }

    void processHeterogeneousGaussianBlurZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
        AlgorithmConcreteKernels::launchHeterogeneousGaussianBlurKernel(frame, processedBuffer_, algoConfig_.blurRadius);
    }

    // CUDA helper functions
    void checkCudaError(cudaError_t err, const std::string& context);
    void allocateCudaMemory(void** devPtr, size_t size, const std::string& context);
    void freeCudaMemory(void* devPtr, const std::string& context);

    // ---------------------- Internal State ----------------------
    std::atomic<bool> running_{false};             ///< Controls the thread run-loop
    std::shared_ptr<SharedQueue<FrameData>> inputQueue_;    ///< Copy-based input
    std::shared_ptr<SharedQueue<FrameData>> outputQueue_;   ///< Copy-based output
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueueZeroCopy_;  ///< Zero-copy input
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> outputQueueZeroCopy_; ///< Zero-copy output

    AlgorithmConfig algoConfig_;                          ///< Current algorithm config
    std::function<void(const std::string&)> errorCallback_;///< Optional error callback

    // Metrics: framesCount_, totalProcTime_, etc.
    mutable std::mutex metricMutex_;
    double fps_           = 0.0;
    double avgProcTime_   = 0.0;
    int    framesCount_   = 0;
    double totalProcTime_ = 0.0;

    std::vector<uint8_t> processedBuffer_;  ///< Buffer used for transformations

    // For last-second-based metrics
    mutable std::atomic<int> framesProcessed_{0};
    mutable std::chrono::steady_clock::time_point lastUpdateTime_;
    mutable double lastFPS_           = 0.0;
    mutable double lastProcessingTime_= 0.0;

    ThreadManager& threadManager_;  ///< Reference to ThreadManager


    // DataConcrete to support SystemMetricsAggregatorImpl injection and push camera metrics in real-time using the aggregator
    std::shared_ptr<ISystemMetricsAggregator> metricAggregator_;

};

// ====================== Implementation ======================

// Constructor Definitions

inline AlgorithmConcrete::AlgorithmConcrete(ThreadManager& threadManager)
    : running_(false)
    , inputQueue_(nullptr)
    , outputQueue_(nullptr)
    , inputQueueZeroCopy_(nullptr)
    , outputQueueZeroCopy_(nullptr)
    , lastUpdateTime_(std::chrono::steady_clock::now())
    , threadManager_(threadManager)
{
    spdlog::warn("[AlgorithmConcrete] Default constructor: no queues provided.");
}

inline AlgorithmConcrete::AlgorithmConcrete(std::shared_ptr<SharedQueue<FrameData>> inputQueue,
                                            std::shared_ptr<SharedQueue<FrameData>> outputQueue,
                                            ThreadManager& threadManager)
    : running_(false)
    , inputQueue_(std::move(inputQueue))
    , outputQueue_(std::move(outputQueue))
    , inputQueueZeroCopy_(nullptr)
    , outputQueueZeroCopy_(nullptr)
    , lastUpdateTime_(std::chrono::steady_clock::now())
    , threadManager_(threadManager)
{
    spdlog::info("[AlgorithmConcrete] Constructor with FrameData queues.");
}

inline AlgorithmConcrete::AlgorithmConcrete(std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
                                            std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> outputQueue,
                                            ThreadManager& threadManager,
                                            std::shared_ptr<ISystemMetricsAggregator> aggregator)
    : running_(false)
    , inputQueue_(nullptr)
    , outputQueue_(nullptr)
    , inputQueueZeroCopy_(std::move(inputQueue))
    , outputQueueZeroCopy_(std::move(outputQueue))
    , lastUpdateTime_(std::chrono::steady_clock::now())
    , threadManager_(threadManager)
    //,algoConfig_(config)
    ,metricAggregator_(std::move(aggregator)) // injected
     {

        if (algoConfig_.algorithmType == AlgorithmType::OpticalFlow_LucasKanade) {
            opticalFlowProcessor_ = std::make_unique<LucasKanadeOpticalFlow>(algoConfig_.opticalFlowConfig);
        }
        spdlog::info("[AlgorithmConcrete] Constructor with ZeroCopy queues.");
    
    // Set up the error callback to log errors
    }

inline AlgorithmConcrete::~AlgorithmConcrete() {
    stopAlgorithm();
}


// ---------------- Factory Methods ----------------

inline std::shared_ptr<IAlgorithm> AlgorithmConcrete::createAlgorithm(
    AlgorithmType type,
    std::shared_ptr<SharedQueue<FrameData>> inputQueue,
    std::shared_ptr<SharedQueue<FrameData>> outputQueue,
    ThreadManager& threadManager)
{
    if (!inputQueue || !outputQueue) {
        spdlog::error("[AlgorithmConcrete] createAlgorithm: FrameData queues must be non-null.");
        return nullptr;
    }
    auto algo = std::make_shared<AlgorithmConcrete>(inputQueue, outputQueue, threadManager); 
    AlgorithmConfig cfg;
    cfg.algorithmType = type;
    algo->configure(cfg);
    return algo;
}

inline std::shared_ptr<IAlgorithm> AlgorithmConcrete::createAlgorithmZeroCopy(
    AlgorithmType type,
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> inputQueue,
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> outputQueue,
    ThreadManager& threadManager,
    std::shared_ptr<ISystemMetricsAggregator> aggregator)
{
    if (!inputQueue || !outputQueue) {
        spdlog::error("[AlgorithmConcrete] createAlgorithmZeroCopy: ZeroCopy queues must be non-null.");
        return nullptr;
    }
    auto algo = std::make_shared<AlgorithmConcrete>(inputQueue, outputQueue, threadManager, aggregator); // Pass the aggregator
    AlgorithmConfig cfg;
    cfg.algorithmType = type;
    algo->configure(cfg);
    return algo;
}

// ---------------- IAlgorithm Overrides ----------------

inline void AlgorithmConcrete::startAlgorithm() {
    if (running_) {
        spdlog::warn("[AlgorithmConcrete] Algorithm is already running.");
        return;
    }
    running_ = true;

    if (inputQueueZeroCopy_ && outputQueueZeroCopy_) {
        // Zero-copy mode
        threadManager_.addThread("AlgorithmProcessingZeroCopy",
            std::thread(&AlgorithmConcrete::threadLoopZeroCopy, this));
    } else if (inputQueue_ && outputQueue_) {
        // Copy-based mode
        threadManager_.addThread("AlgorithmProcessing",
            std::thread(&AlgorithmConcrete::threadLoop, this));
    } else {
        reportError("[AlgorithmConcrete] No valid queues configured for startAlgorithm().");
        running_ = false;
        return;
    }
    spdlog::info("[AlgorithmConcrete] Algorithm thread started, mode: {}.",
                 algorithmTypeToString(algoConfig_.algorithmType));
}

inline void AlgorithmConcrete::stopAlgorithm() {
    if (!running_) {
        return;
    }
    running_ = false;

    // Optionally stop the output queues to unblock any waiting push
    if (outputQueue_) {
        outputQueue_->stop();
    }
    if (outputQueueZeroCopy_) {
        outputQueueZeroCopy_->stop();
    }

    // Join threads
    threadManager_.joinThreadsFor("AlgorithmProcessing");
    threadManager_.joinThreadsFor("AlgorithmProcessingZeroCopy");

    spdlog::info("[AlgorithmConcrete] Algorithm thread stopped.");
}

inline bool AlgorithmConcrete::configure(const AlgorithmConfig& config) {
    algoConfig_ = config;

    if (algoConfig_.algorithmType == AlgorithmType::OpticalFlow_LucasKanade) {
        opticalFlowProcessor_ = std::make_unique<LucasKanadeOpticalFlow>(algoConfig_.opticalFlowConfig);
    }
    
    spdlog::info("[AlgorithmConcrete] Configured algorithm: {}",
                 algorithmTypeToString(config.algorithmType));
    return true;
}

inline void AlgorithmConcrete::setErrorCallback(std::function<void(const std::string&)> callback) {
    errorCallback_ = std::move(callback);
}

inline std::tuple<double, double> AlgorithmConcrete::getAlgorithmMetrics() const {
    // Return (FPS, lastProcessingTime) – you can also return avgProcTime_ if you prefer
    return {getLastFPS(), lastProcessingTime_};
}

inline double AlgorithmConcrete::getLastFPS() const {
    return lastFPS_;
}

inline double AlgorithmConcrete::getFps() const {
    // For example, just return lastFPS_. Could do something else if you prefer
    return lastFPS_;
}

inline double AlgorithmConcrete::getAverageProcTime() const {
    return avgProcTime_;
}

inline const uint8_t* AlgorithmConcrete::getProcessedBuffer() const {
    return processedBuffer_.data();
}

// ---------------- Thread Loops ----------------

inline void AlgorithmConcrete::threadLoop() {
    auto tStart = std::chrono::steady_clock::now();
    while (running_) {
        FrameData inputFrame;
        FrameData outputFrame;

        if (!inputQueue_->pop(inputFrame)) {
            // Queue empty, yield briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Process the frame
        if (processFrame(inputFrame, outputFrame)) {
            spdlog ::debug("[AlgorithmConcrete] Processing successful: frameNumber={}",
                     inputFrame.frameNumber);

            outputQueue_->push(outputFrame);
            
            spdlog ::debug("[AlgorithmConcrete] Output pushed: frameNumber={}",
                     outputFrame.frameNumber);


            framesCount_++;

            double elapsedSec = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tStart).count();
            if (elapsedSec >= 1.0) {
                updateMetrics(elapsedSec);
                tStart = std::chrono::steady_clock::now();
                framesCount_ = 0;
                totalProcTime_ = 0.0;
            }
        }
        if (metricAggregator_) {
            // metricAggregator_->pushMetrics(std::chrono::system_clock::now(), [inferenceTimeMs](SystemMetricsSnapshot& snapshot) {
            //     snapshot.algorithmStats.inferenceTimeMs = inferenceTimeMs;
            // });

            metricAggregator_->pushMetrics(std::chrono::system_clock::now(), [&](SystemMetricsSnapshot& snapshot) {
                snapshot.algorithmStats.inferenceTimeMs = lastProcessingTime_;
            });
        }
        

    }
    spdlog::info("[AlgorithmConcrete] Exiting FrameData thread loop ({}).",
                 algorithmTypeToString(algoConfig_.algorithmType));
}

inline void AlgorithmConcrete::threadLoopZeroCopy() {
    auto tStart = std::chrono::steady_clock::now();
    while (running_) {
        std::shared_ptr<ZeroCopyFrameData> inputFrame;
        std::shared_ptr<ZeroCopyFrameData> outputFrame;

        if (!inputQueueZeroCopy_->pop(inputFrame)) {
            // Queue empty, yield briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Process the zero-copy frame
        if (processFrameZeroCopy(inputFrame, outputFrame)) {
            spdlog ::debug("[AlgorithmConcrete] ZeroCopy processing successful: frameNumber={}",
                     inputFrame->frameNumber);
            
            outputQueueZeroCopy_->push(outputFrame);
            
            spdlog::debug("[AlgorithmConcrete] ZeroCopy output pushed: frameNumber={}",
                     outputFrame->frameNumber);
            
            framesCount_++;

            double elapsedSec = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tStart).count();
            if (elapsedSec >= 1.0) {
                updateMetrics(elapsedSec);
                tStart = std::chrono::steady_clock::now();
                framesCount_ = 0;
                totalProcTime_ = 0.0;
            }
        }
    }
    spdlog::info("[AlgorithmConcrete] Exiting ZeroCopy thread loop ({}).",
                 algorithmTypeToString(algoConfig_.algorithmType));
}

// ---------------- Processing Functions (FrameData) ----------------

inline bool AlgorithmConcrete::processFrame(const FrameData& inputFrame, FrameData& outputFrame) {
    if (inputFrame.dataVec.empty() || inputFrame.size == 0) {
        spdlog::warn("[AlgorithmConcrete] Received empty or invalid FrameData.");
        return false;
    }

    auto start = std::chrono::steady_clock::now();

    // Prepare output
    outputFrame.frameNumber = inputFrame.frameNumber;
    outputFrame.width       = inputFrame.width;
    outputFrame.height      = inputFrame.height;
    outputFrame.size        = inputFrame.size;
    outputFrame.dataVec.resize(inputFrame.size);

    // Run the chosen algorithm
    switch (algoConfig_.algorithmType) {
        case AlgorithmType::Invert:
            processInvert(inputFrame);
            break;
        case AlgorithmType::Grayscale:
            processGrayscale(inputFrame);
            break;
        case AlgorithmType::EdgeDetection:
            processEdgeDetection(inputFrame);
            break;
        case AlgorithmType::GaussianBlur:
            processGaussianBlur(inputFrame);
            break;
        case AlgorithmType::MatrixMultiply:
            processMatrixMultiply();
            break;
        case AlgorithmType::Mandelbrot:
            processMandelbrot();
            break;
        case AlgorithmType::PasswordHash:
            processPasswordHash(inputFrame);
            break;
        case AlgorithmType::MultiPipeline:
            processMultiPipeline(inputFrame);
            break;
        case AlgorithmType::GPUMatrixMultiply:
            processGPUMatrixMultiply();
            break;
        case AlgorithmType::MultiThreadedInvert:
            processMultiThreadedInvert(inputFrame);
            break;
        default:
            spdlog::warn("[AlgorithmConcrete] Unknown algorithm type in processFrame().");
            return false;
    }

    // Move processed data into output
    outputFrame.dataVec = processedBuffer_;

    auto end = std::chrono::steady_clock::now();
    lastProcessingTime_ = std::chrono::duration<double, std::milli>(end - start).count();
    totalProcTime_     += lastProcessingTime_;
    return true;
}

// ---------------- Processing Functions (ZeroCopyFrameData) ----------------

inline bool AlgorithmConcrete::processFrameZeroCopy(
    const std::shared_ptr<ZeroCopyFrameData>& inputFrame,
    std::shared_ptr<ZeroCopyFrameData>& outputFrame)
{
    if (!inputFrame || !inputFrame->dataPtr || inputFrame->size == 0) {
        spdlog::error("[AlgorithmConcrete] Invalid zero-copy input frame.");
        return false;
    }

    auto start = std::chrono::steady_clock::now();

    outputFrame = std::make_shared<ZeroCopyFrameData>(*inputFrame);
    outputFrame->dataPtr = malloc(inputFrame->size);
    if (!outputFrame->dataPtr) {
        spdlog::error("[AlgorithmConcrete] Failed to allocate output buffer.");
        return false;
    }

    // Potentially do in-place modification:
    //outputFrame = inputFrame;

    // Switch on the config
    switch (algoConfig_.algorithmType) {
        case AlgorithmType::Invert:
            processInvertZeroCopy(inputFrame);
            break;
        case AlgorithmType::Grayscale:
            processGrayscaleZeroCopy(inputFrame);
            break;
        case AlgorithmType::EdgeDetection:
            processEdgeDetectionZeroCopy(inputFrame);
            break;
        case AlgorithmType::GaussianBlur:
            processGaussianBlurZeroCopy(inputFrame);
            break;
        case AlgorithmType::MatrixMultiply:
            processMatrixMultiplyZeroCopy();
            break;
        case AlgorithmType::Mandelbrot:
            processMandelbrotZeroCopy();
            break;
        case AlgorithmType::PasswordHash:
            processPasswordHashZeroCopy(inputFrame);
            break;
        case AlgorithmType::MultiPipeline:
            processMultiPipelineZeroCopy(inputFrame);
            break;
        case AlgorithmType::GPUMatrixMultiply:
            processGPUMatrixMultiplyZeroCopy();
            break;
        case AlgorithmType::MultiThreadedInvert:
            processMultiThreadedInvertZeroCopy(inputFrame);
            break;
        case AlgorithmType::OpticalFlow_LucasKanade:
            processOpticalFlow(inputFrame);
            break;
        case AlgorithmType::SobelEdge: processSobelEdgeZeroCopy(inputFrame); break;
        case AlgorithmType::MedianFilter: processMedianFilterZeroCopy(inputFrame); break;
        case AlgorithmType::HistogramEqualization: processHistogramEqualizationZeroCopy(inputFrame); break;
        case AlgorithmType::HeterogeneousGaussianBlur: processHeterogeneousGaussianBlurZeroCopy(inputFrame); break;
        default:
            spdlog::warn("[AlgorithmConcrete] Unknown algorithm type in processFrameZeroCopy().");
            return false;
    }
    
    memcpy(outputFrame->dataPtr, processedBuffer_.data(), inputFrame->size); // Copy processed data to output
    auto end = std::chrono::steady_clock::now();
    lastProcessingTime_ = std::chrono::duration<double, std::milli>(end - start).count();
    totalProcTime_     += lastProcessingTime_;
    return true;
}

// ---------------- Metric Helpers ----------------

// MOD: Fixed metrics aggregation
inline void AlgorithmConcrete::updateMetrics(double elapsedSec) {
    fps_ = static_cast<double>(framesCount_) / elapsedSec;
    if (framesCount_ > 0) {
        avgProcTime_ = totalProcTime_ / framesCount_;
    } else {
        avgProcTime_ = 0.0;
    }
    lastFPS_ = fps_;

    spdlog::info("[AlgorithmConcrete] FPS: {:.1f}, AvgTime: {:.2f} ms, Type: {}",
                 fps_, avgProcTime_, algorithmTypeToString(algoConfig_.algorithmType));

    if (metricAggregator_) {
        metricAggregator_->pushMetrics(std::chrono::system_clock::now(), 
        [this](SystemMetricsSnapshot& snapshot) {
            snapshot.algorithmStats.fps= fps_;
            snapshot.algorithmStats.avgProcTimeMs= avgProcTime_;    //avgProcTimeMs
            // Add GPU-specific metrics
            cudaMemGetInfo(&snapshot.algorithmStats.gpuFreeMemory, &snapshot.algorithmStats.gpuTotalMemory);
            });

    //PerformanceLogger::getInstance().pushAlgorithmStats(fps_, avgProcTime_);
    //metricAggregator_->pushMetrics(snapshot.algorithmStats.fps_, snapshot.algorithmStats.avgProcTime_);
    }
}
// ---------------- Utility & ParallelFor ----------------

inline std::string AlgorithmConcrete::algorithmTypeToString(AlgorithmType type) const {
    switch (type) {
        case AlgorithmType::Invert:             return "Invert";
        case AlgorithmType::Grayscale:          return "Grayscale";
        case AlgorithmType::EdgeDetection:      return "EdgeDetection";
        case AlgorithmType::GaussianBlur:       return "GaussianBlur";
        case AlgorithmType::MatrixMultiply:     return "MatrixMultiply";
        case AlgorithmType::Mandelbrot:         return "Mandelbrot";
        case AlgorithmType::PasswordHash:       return "PasswordHash";
        case AlgorithmType::MultiPipeline:      return "MultiPipeline";
        case AlgorithmType::GPUMatrixMultiply:  return "GPUMatrixMultiply";
        case AlgorithmType::MultiThreadedInvert:return "MultiThreadedInvert";
        case AlgorithmType::OpticalFlow_LucasKanade: return "OpticalFlow_LucasKanade";
        case AlgorithmType::SobelEdge: return "SobelEdge";
        case AlgorithmType::MedianFilter: return "MedianFilter";
        case AlgorithmType::HistogramEqualization: return "HistogramEqualization";
        case AlgorithmType::HeterogeneousGaussianBlur: return "HeterogeneousGaussianBlur";
        default:                                return "Unknown";
    }
}

inline void AlgorithmConcrete::reportError(const std::string& msg) {
    if (errorCallback_) {
        errorCallback_(msg);
    } else {
        spdlog::error("[AlgorithmConcrete] {}", msg);
    }
}

inline void AlgorithmConcrete::parallelFor(size_t start, size_t end, std::function<void(size_t)> func) {
    const size_t numThreads = algoConfig_.concurrencyLevel;
    if (numThreads <= 1) {
        // Single-threaded fallback
        for (size_t i = start; i < end; i++) {
            func(i);
        }
        return;
    }

    const size_t chunkSize = (end - start + numThreads - 1) / numThreads;
    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    for (size_t t = 0; t < numThreads; ++t) {
        size_t s = start + t * chunkSize;
        if (s >= end) break;  // No more tasks
        size_t e = std::min(end, s + chunkSize);

        workers.emplace_back([=] {
            for (size_t j = s; j < e; j++) {
                func(j);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }
}



// ---------------- Example Processing (Copy-based) ----------------

inline void AlgorithmConcrete::processInvert(const FrameData& frame) {
    processedBuffer_.resize(frame.size);
    parallelFor(0, frame.size, [&](size_t i) {
        processedBuffer_[i] = ~frame.dataVec[i];
    });
}

inline void AlgorithmConcrete::processGrayscale(const FrameData& frame) {
    processedBuffer_.resize(frame.size);
    for (size_t i = 0; i < frame.size; i += 2) {
        processedBuffer_[i]     = frame.dataVec[i];   // Y
        processedBuffer_[i + 1] = 128;                // U/V forced
    }
}

inline void AlgorithmConcrete::processEdgeDetection(const FrameData& frame) {
    processedBuffer_.resize(frame.size);
    int width  = frame.width;
    int height = frame.height;

    // Example "edge" logic
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 2) {
            int idx = y * width * 2 + x * 2;
            if (x > 0 && x < (width - 2)) {
                int grad = std::abs(frame.dataVec[idx] - frame.dataVec[idx - 2]);
                processedBuffer_[idx]     = (uint8_t)grad;
                processedBuffer_[idx + 1] = 128;
            } else {
                processedBuffer_[idx]     = frame.dataVec[idx];
                processedBuffer_[idx + 1] = frame.dataVec[idx + 1];
            }
        }
    }
}

inline void AlgorithmConcrete::processGaussianBlur(const FrameData& frame) {
    const int radius = algoConfig_.blurRadius;
    const int width  = frame.width;
    const int height = frame.height;
    processedBuffer_.resize(frame.size);

    std::vector<uint8_t> temp(frame.size); // intermediate

    // Horizontal pass
    parallelFor(0, height, [&](size_t row) {
        for (int x = 0; x < width; x++) {
            float sum  = 0.f;
            float wsum = 0.f;
            for (int dx = -radius; dx <= radius; dx++) {
                int xx = std::clamp(x + dx, 0, width - 1);
                float w = std::exp(-(dx * dx)/(2.f * radius * radius));
                sum  += frame.dataVec[row * width * 2 + xx * 2] * w;
                wsum += w;
            }
            temp[row * width * 2 + x * 2]     = (uint8_t)(sum / wsum);
            temp[row * width * 2 + x * 2 + 1] = 128;
        }
    });

    // Vertical pass
    parallelFor(0, width, [&](size_t col) {
        for (int y = 0; y < height; y++) {
            float sum  = 0.f;
            float wsum = 0.f;
            for (int dy = -radius; dy <= radius; dy++) {
                int yy = std::clamp(y + dy, 0, height - 1);
                float w = std::exp(-(dy * dy)/(2.f * radius * radius));
                sum  += temp[yy * width * 2 + col * 2] * w;
                wsum += w;
            }
            processedBuffer_[y * width * 2 + col * 2]     = (uint8_t)(sum / wsum);
            processedBuffer_[y * width * 2 + col * 2 + 1] = 128;
        }
    });
}

inline void AlgorithmConcrete::processMatrixMultiply() {
    // Example
    int N = algoConfig_.matrixSize;
    std::vector<float> A(N*N, 1.f);
    std::vector<float> B(N*N, 2.f);
    std::vector<float> C(N*N, 0.f);

    parallelFor(0, N, [&](size_t i) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    });

    processedBuffer_.resize(N*N);
    for (int idx = 0; idx < N*N; idx++) {
        float val = C[idx] * 0.01f;
        processedBuffer_[idx] = (uint8_t)std::min(255.f, val);
    }
}

inline void AlgorithmConcrete::processMandelbrot() {
    // Example
    int width  = 256;
    int height = 256;
    processedBuffer_.resize(width * height * 2, 0);

    parallelFor(0, height, [&](size_t py) {
        float y0 = (py - height/2.0f) * 4.0f / width;
        for (int px = 0; px < width; px++) {
            float x0 = (px - width/2.0f) * 4.0f / width;
            float x = 0.f, y = 0.f;
            int iteration = 0;
            int maxIter   = algoConfig_.mandelbrotIter;

            while (x*x + y*y < 4 && iteration < maxIter) {
                float xtemp = x*x - y*y + x0;
                y = 2*x*y + y0;
                x = xtemp;
                iteration++;
            }
            int index = py*width*2 + px*2;
            processedBuffer_[index]   = (uint8_t)((iteration*255)/maxIter);
            processedBuffer_[index+1] = 128;
        }
    });
}

inline void AlgorithmConcrete::processPasswordHash(const FrameData& frame) {
    processedBuffer_.resize(frame.size);
    int iterations = 10000;

    parallelFor(0, frame.size, [&](size_t i) {
        uint32_t hash = (uint32_t)frame.dataVec[i];
        for (int j = 0; j < iterations; j++) {
            hash = (hash << 5) + hash + j;
        }
        processedBuffer_[i] = (uint8_t)(hash % 256);
    });
}

inline void AlgorithmConcrete::processMultiPipeline(const FrameData& frame) {
    processedBuffer_.resize(frame.size);
    // Example pipeline: invert + force chroma
    for (size_t i = 0; i < frame.size; i++) {
        processedBuffer_[i] = ~frame.dataVec[i];
    }
    for (size_t i = 0; i < frame.size; i += 2) {
        processedBuffer_[i+1] = 128;
    }
}

inline void AlgorithmConcrete::processGPUMatrixMultiply() {
    spdlog::warn("[AlgorithmConcrete] GPUMatrixMultiply not implemented; returning dummy data.");
    processedBuffer_.resize(512, 128);
}

inline void AlgorithmConcrete::processMultiThreadedInvert(const FrameData& frame) {
    processedBuffer_.resize(frame.size);
    size_t chunkSize = frame.size / algoConfig_.concurrencyLevel;
    std::vector<std::future<void>> futures;
    futures.reserve(algoConfig_.concurrencyLevel);

    for (int i = 0; i < algoConfig_.concurrencyLevel; i++) {
        size_t start = i * chunkSize;
        size_t end   = (i == algoConfig_.concurrencyLevel-1) ? frame.size : (start + chunkSize);
        futures.push_back(std::async(std::launch::async, [this, &frame, start, end] {
            for (size_t idx = start; idx < end; idx++) {
                processedBuffer_[idx] = ~frame.dataVec[idx];
            }
        }));
    }
    for (auto& f : futures) {
        f.get();
    }
}

// ---------------- Example Processing (ZeroCopyFrameData) ----------------

inline void AlgorithmConcrete::processInvertZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
    if (!frame || !frame->dataPtr) return;
    uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
    parallelFor(0, frame->size, [&](size_t i) {data[i] = ~data[i];});

    //parallelFor(0, frame->size, [&](size_t i) { data[i] = ~data[i]; });
}

inline void AlgorithmConcrete::processGrayscaleZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
    // Show an example where we do a read->write into processedBuffer_
    processedBuffer_.resize(frame->size);
    uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);

    for (size_t i = 0; i < frame->size; i += 2) {
        processedBuffer_[i] = data[i];
        processedBuffer_[i+1] = 128;
    }

    // Could optionally copy back to data if you want in-place
    std::memcpy(data, processedBuffer_.data(), frame->size);
}

inline void AlgorithmConcrete::processEdgeDetectionZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
    processedBuffer_.resize(frame->size);
    uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
    int width  = frame->width;
    int height = frame->height;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 2) {
            int idx = y*width*2 + x*2;
            if (x>0 && x<(width-2)) {
                int grad = std::abs(data[idx] - data[idx-2]);
                processedBuffer_[idx]     = (uint8_t)grad;
                processedBuffer_[idx+1]   = 128;
            } else {
                processedBuffer_[idx]     = data[idx];
                processedBuffer_[idx+1]   = data[idx+1];
            }
        }
    }
    std::memcpy(data, processedBuffer_.data(), frame->size);
}

inline void AlgorithmConcrete::processGaussianBlurZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
    const int radius = algoConfig_.blurRadius;
    const int width  = frame->width;
    const int height = frame->height;

    processedBuffer_.resize(frame->size);
    std::vector<uint8_t> temp(frame->size);

    uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);

    // Horizontal pass
    parallelFor(0, height, [&](size_t y) {
        for (int x = 0; x < width; x++) {
            float sum = 0.f;
            float wsum= 0.f;
            for (int dx = -radius; dx <= radius; dx++) {
                int xx = std::clamp(x + dx, 0, width-1);
                float w = std::exp(-(dx*dx)/(2.f*radius*radius));
                sum  += data[y*width*2 + xx*2] * w;
                wsum += w;
            }
            temp[y*width*2 + x*2]     = (uint8_t)(sum/wsum);
            temp[y*width*2 + x*2 + 1] = 128;
        }
    });

    // Vertical pass
    parallelFor(0, width, [&](size_t col) {
        for (int y = 0; y < height; y++) {
            float sum = 0.f;
            float wsum= 0.f;
            for (int dy = -radius; dy <= radius; dy++) {
                int yy = std::clamp(y + dy, 0, height-1);
                float w = std::exp(-(dy*dy)/(2.f*radius*radius));
                sum  += temp[yy*width*2 + col*2] * w;
                wsum += w;
            }
            processedBuffer_[y*width*2 + col*2]     = (uint8_t)(sum/wsum);
            processedBuffer_[y*width*2 + col*2 + 1] = 128;
        }
    });

    // Copy back to data
    std::memcpy(data, processedBuffer_.data(), frame->size);
}

inline void AlgorithmConcrete::processMatrixMultiplyZeroCopy() {
    // Example
    int N = algoConfig_.matrixSize;
    std::vector<float> A(N*N, 1.f);
    std::vector<float> B(N*N, 2.f);
    std::vector<float> C(N*N, 0.f);

    parallelFor(0, N, [&](size_t i) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    });

    processedBuffer_.resize(N*N);
    for (int idx = 0; idx < N*N; idx++) {
        float val = C[idx]*0.01f;
        processedBuffer_[idx] = (uint8_t)std::min(255.f, val);
    }
}

inline void AlgorithmConcrete::processMandelbrotZeroCopy() {
    int width = 256;
    int height= 256;
    processedBuffer_.resize(width*height*2, 0);

    parallelFor(0, height, [&](size_t py) {
        float y0 = (py - height/2.0f)*4.0f/width;
        for (int px = 0; px < width; px++) {
            float x0 = (px - width/2.0f)*4.0f/width;
            float x = 0.f, y = 0.f;
            int iteration= 0;
            int maxIter  = algoConfig_.mandelbrotIter;

            while (x*x + y*y < 4 && iteration<maxIter) {
                float xtemp = x*x - y*y + x0;
                y = 2*x*y + y0;
                x = xtemp;
                iteration++;
            }
            int idx = py*width*2 + px*2;
            processedBuffer_[idx]   = (uint8_t)((iteration*255)/maxIter);
            processedBuffer_[idx+1] = 128;
        }
    });
    // If you want to apply it in-place to the frame->dataPtr, do so
}

inline void AlgorithmConcrete::processPasswordHashZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
    processedBuffer_.resize(frame->size);
    uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);
    int iterations = 10000;

    parallelFor(0, frame->size, [&](size_t i) {
        uint32_t hash = (uint32_t)data[i];
        for (int j = 0; j < iterations; j++) {
            hash = (hash<<5) + hash + j;
        }
        processedBuffer_[i] = (uint8_t)(hash % 256);
    });

    // Copy back
    std::memcpy(data, processedBuffer_.data(), frame->size);
}

inline void AlgorithmConcrete::processMultiPipelineZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
    processedBuffer_.resize(frame->size);
    uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);

    // Example pipeline: invert + force chroma
    for (size_t i = 0; i < frame->size; i++) {
        processedBuffer_[i] = ~data[i];
    }
    for (size_t i = 0; i < frame->size; i += 2) {
        processedBuffer_[i + 1] = 128;
    }

    // Copy back
    std::memcpy(data, processedBuffer_.data(), frame->size);
}

inline void AlgorithmConcrete::processGPUMatrixMultiplyZeroCopy() {
    spdlog::warn("[AlgorithmConcrete] GPUMatrixMultiplyZeroCopy not implemented.");
    processedBuffer_.resize(512, 128);
}

inline void AlgorithmConcrete::processMultiThreadedInvertZeroCopy(const std::shared_ptr<ZeroCopyFrameData>& frame) {
    processedBuffer_.resize(frame->size);
    uint8_t* data = static_cast<uint8_t*>(frame->dataPtr);

    size_t chunkSize = frame->size / algoConfig_.concurrencyLevel;
    std::vector<std::future<void>> futures;
    futures.reserve(algoConfig_.concurrencyLevel);

    for (int i = 0; i < algoConfig_.concurrencyLevel; i++) {
        size_t start = i * chunkSize;
        size_t end   = (i == algoConfig_.concurrencyLevel-1) ? frame->size : (start + chunkSize);
        futures.push_back(std::async(std::launch::async, [this, data, start, end] {
            for (size_t idx = start; idx < end; idx++) {
                processedBuffer_[idx] = ~data[idx];
            }
        }));
    }
    for (auto& f : futures) {
        f.get();
    }
    std::memcpy(data, processedBuffer_.data(), frame->size);
}



//  OpticalFlow  
// MOD: Implemented optical flow processing
void AlgorithmConcrete::processOpticalFlow(const std::shared_ptr<ZeroCopyFrameData>& frame) {
    //spdlog::info("Algorithm = optical flow processing: Step 01");
    if (!frame || !frame->dataPtr) {
        spdlog::error("[AlgorithmConcrete] Invalid zero-copy input frame.");
        return;
    }
    //auto start = std::chrono::steady_clock::now();
    // Prepare output
    std::shared_ptr<ZeroCopyFrameData> outputFrame = std::make_shared<ZeroCopyFrameData>();
    // Process the optical flow using the previous frame and the current frame
    //spdlog::info("Algorithm = optical flow processing: Step 02");
    if (previousFrame_) {
        std::shared_ptr<ZeroCopyFrameData> flowOutput;
       // spdlog::info("Algorithm = optical flow processing: Step 03");
        opticalFlowProcessor_->computeOpticalFlow(previousFrame_, frame, flowOutput);
        if (outputQueueZeroCopy_) {
            outputQueueZeroCopy_->push(flowOutput);
        }
        
    }
    previousFrame_ = frame;
    //spdlog::info("Algorithm = optical flow processing: Step 04");
}

// MOD: Implemented Sobel edge detection


inline void AlgorithmConcrete::checkCudaError(cudaError_t err, const std::string& context) {
    if (err != cudaSuccess) {
        std::string msg = "[AlgorithmConcrete] CUDA error in " + context + ": " + cudaGetErrorString(err);
        reportError(msg);
    }
}

inline void AlgorithmConcrete::allocateCudaMemory(void** devPtr, size_t size, const std::string& context) {
    checkCudaError(cudaMalloc(devPtr, size), context);
}

inline void AlgorithmConcrete::freeCudaMemory(void* devPtr, const std::string& context) {
    checkCudaError(cudaFree(devPtr), context);
}

// ================= Dynamic Runtime Algorithm Switching ==================

/**
 * @brief Allows changing the algorithm type at runtime dynamically
 * @param newType The new AlgorithmType to switch to
 */
inline void AlgorithmConcrete::setAlgorithmType(AlgorithmType newType) {
    std::lock_guard<std::mutex> lock(metricMutex_);  // protect algoConfig_
    
    if (algoConfig_.algorithmType == newType) {
        spdlog::info("[AlgorithmConcrete] Algorithm type already set to {}. No change.", algorithmTypeToString(newType));
        return;
    }

    spdlog::info("[AlgorithmConcrete] Changing algorithm from {} to {}.",
                 algorithmTypeToString(algoConfig_.algorithmType),
                 algorithmTypeToString(newType));

    algoConfig_.algorithmType = newType;

    // Reinitialize any specialized resources if needed
    if (newType == AlgorithmType::OpticalFlow_LucasKanade) {
        opticalFlowProcessor_ = std::make_unique<LucasKanadeOpticalFlow>(algoConfig_.opticalFlowConfig);
        previousFrame_.reset();  // Clear previous frame because optical flow requires new history
    } else {
        opticalFlowProcessor_.reset();  // If not OpticalFlow, clean the processor
    }
}


///=====================TESTING  code Final Version ==============================