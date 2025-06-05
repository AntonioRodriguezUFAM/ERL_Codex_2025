 // SystemCaptureFactory.h

#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <chrono>
#include <tuple>
#include <spdlog/spdlog.h>
#include <SDL2/SDL.h>

#include "../Interfaces/IData.h"
#include "../Interfaces/IAlgorithm.h"
#include "../Interfaces/ISoC.h"
#include "../Interfaces/ISystemProfiling.h"

#include "../SharedStructures/FrameData.h"
#include "../SharedStructures/CameraConfig.h"
#include "../SharedStructures/AlgorithmConfig.h"
#include "../SharedStructures/SoCConfig.h"
#include "../SharedStructures/DisplayConfig.h"
#include "../SharedStructures/SharedQueue.h"
#include "../SharedStructures/jetsonNanoInfo.h"

#include "../Concretes/SdlDisplayConcrete.h"
#include "../Concretes/DataConcrete.h"
#include "SystemModellingFactory.h"

/**
 * @class SystemCaptureFactory
 * @brief Coordinates the main “capture system” in an asynchronous pipeline:
 *   - Multiple cameras (async streaming)
 *   - A shared frame queue (producer-consumer)
 *   - A separate algorithm thread pulling from the queue
 *   - A display thread (optional)
 *   - A SoC monitoring thread
 *
 * Implements ISystemProfiling to provide SoC, camera, and algorithm metrics 
 * (queried by e.g. SystemProfilingFactory).
 *
 * Usage:
 *   1. Construct with pointers to SoC, Algorithm, and camera(s).
 *   2. Call initializeCapture(cameraConfig, algoConfig) to open devices, configure modules, and start threads.
 *   3. Each camera pushes frames into frameQueue_ (captureLoop).
 *   4. The algorithm thread pops frames (algorithmLoop).
 *   5. SoC monitoring logs SoC stats (SoCLoop).
 *   6. Optionally, a display thread (displayLoop).
 */
class SystemCaptureFactory : public ISystemProfiling {
public:
    // Constructors for 1-camera or 2-camera setups.
    SystemCaptureFactory(std::shared_ptr<ISoC> soc,
                         std::shared_ptr<IAlgorithm> algo,
                         std::shared_ptr<IData> camera0,
                         std::shared_ptr<IData> camera1);

    SystemCaptureFactory(std::shared_ptr<ISoC> soc,
                         std::shared_ptr<IAlgorithm> algo,
                         std::shared_ptr<IData> camera);

    ~SystemCaptureFactory() override;

    // ===========================
    // ISystemProfiling interface
    // ===========================
    JetsonNanoInfo getSoCMetrics() const override;
    std::tuple<double, int> getCameraMetrics(int cameraIndex)  const override;
    std::tuple<double, double> getAlgorithmMetrics() const override;

      // ===========================
    // System control methods
    // ===========================
    /**
     * @brief Initialize capture pipeline (cameras, SoC, display, and threads).
     *        Called once before streaming.
     * @param cameraConfig Camera parameters (width, height, pixelFormat, fps).
     * @param algoConfig   Algorithm parameters (e.g., concurrency, model path).
     */

  void initializeCapture(const CameraConfig& cameraConfig, const AlgorithmConfig& algoConfig);

    /**
     * @brief Stop capture pipeline (joins threads, closes devices).
     */
    void stopCapture();

    /**
     * @brief Reset devices and free resources (after stopCapture).
     */
    void resetDevices();

    /**
     * @brief Temporarily pause the entire pipeline (capture, algorithm, display).
     */
    void pauseAll();

    /**
     * @brief Resume after pauseAll().
     */
    void resumeAll();

    /**
     * @brief Optional: Sets a display to show original/processed frames.
     */
    void setDisplay(std::shared_ptr<IDisplay> display);
    
    // ===========================
    // Configuration methods
    // ===========================
    /**
     * @brief Re-configure a specific camera with new parameters.
     */
    void configureCamera(size_t index, const CameraConfig& cfg);

    /**
     * @brief Re-configure the algorithm with new parameters.
     */
    void configureAlgorithm(const AlgorithmConfig& cfg);

    /**
     * @brief Re-configure the SoC monitoring.
     */
    void configureSoC(const SoCConfig& cfg);

    /**
     * @brief Re-configure the display (width, height, additional parameters).
     */
    void configureDisplay(const DisplayConfig& cfg);
    // ===========================
    // Error handling
    // ===========================
    /**
     * @brief Set a global error callback for reporting all errors in this system.
     */
    void setGlobalErrorCallback(std::function<void(const std::string&)>) ;
    
    // ===========================
    // Capture Metrics (getters)
    // ===========================
    std::shared_ptr<ISoC> getSoC() const;
    std::shared_ptr<IData> getCamera(int index = 0) const;
    std::shared_ptr<IAlgorithm> getAlgorithm() const;



private:
    // ===========================
    // Thread routines
    // ===========================
    // Internal threads for each subsystem
    void captureLoop(IData* camera);         // Producer loop for each camera
    void SoCLoop(ISoC* soc);             // SoC monitoring
    void algorithmLoop(IAlgorithm* alg);     // Consumer loop
    void displayLoop();                      // Optional UI thread

    // ===========================
    // Utility
    // ===========================
    void reportError(const std::string& msg);

private:
    // Control flags
    std::atomic<bool> running_;
    std::atomic<bool> paused_;
    
    // Subsystems
    std::shared_ptr<ISoC> soc_;
    std::shared_ptr<IAlgorithm> algo_;
    std::vector<std::shared_ptr<IData>> cameras_; // If you support multiple cameras
    std::shared_ptr<IDisplay> display_;

    // Threads for each subsystem
    std::vector<std::thread> cameraThreads_;
    std::thread socThread_;
    std::thread displayThread_;
    std::thread algorithmThread_;

    // Producer-Consumer queue for frames
    SharedQueue<FrameData> frameQueue_;

    // Global error callback
    std::function<void(const std::string&)> errorCallback_;

    // Display resolution (can match cameraConfig or be distinct)
    int displayWidth_  =320; //640; //320;
    int displayHeight_ =240;// 480; //240;

    // Example max queue size to avoid unbounded memory usage
    static constexpr size_t MAX_QUEUE_SIZE = 10;

    // For any processed frame data if needed
    // For any processed frame data if needed
    std::mutex processedMutex_;
    std::vector<uint8_t> processedRGB_; // For storing any processed output
    std::vector<uint8_t> rgbBuffer_;    // For conversion from YUYV → RGB

    // Configuration settings
    // Store last cameraConfig for reference
    CameraConfig cameraConfig_;


};

// ================================================
// Implementation (typically in SystemCaptureFactory.cpp)
// ================================================

// Constructor with 2 cameras
inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
                                                  std::shared_ptr<IAlgorithm> algo,
                                                  std::shared_ptr<IData> camera0,
                                                  std::shared_ptr<IData> camera1)
    : soc_(std::move(soc))
    , algo_(std::move(algo))
    , running_(false)
    , paused_(false)
    , errorCallback_(nullptr)
{
    if (!camera0 || !camera1) {
        throw std::invalid_argument("[SystemCaptureFactory] Null camera pointer(s).");
    }
    cameras_.push_back(std::move(camera0));
    cameras_.push_back(std::move(camera1));

    if (cameras_.empty()) {
        throw std::invalid_argument("[SystemCaptureFactory] At least one camera is required.");
    }
}

// Constructor with 1 camera
inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
                                                  std::shared_ptr<IAlgorithm> algo,
                                                  std::shared_ptr<IData> camera)
    : soc_(std::move(soc))
    , algo_(std::move(algo))
    , running_(false)
    , paused_(false)
    , errorCallback_(nullptr)
    {
    if (!camera) {
            throw std::invalid_argument("[SystemCaptureFactory] Null camera pointer given.");
        }
        cameras_.push_back(std::move(camera));
        if (cameras_.empty()) {
            throw std::invalid_argument("[SystemCaptureFactory] At least one camera is required.");
        }
    }
// Destructor
inline SystemCaptureFactory::~SystemCaptureFactory() {
    stopCapture();
}

// ===========================
// SystemCaptureFactory Metrics
// ===========================

// ===========================
// ISystemProfiling interface
// ===========================
inline JetsonNanoInfo SystemCaptureFactory::getSoCMetrics() const {
    return soc_ ? soc_->getPerformance() : JetsonNanoInfo{};
}

// ===========================
// Camera Metrics
// ===========================
inline std::tuple<double, int> SystemCaptureFactory::getCameraMetrics(int cameraIndex) const {
    if (cameraIndex < 0 || static_cast<size_t>(cameraIndex) >= cameras_.size()) {
        std::cout << "Invalid camera index: "<<cameraIndex<< std::endl;
//        reportError("Invalid camera index: " + std::to_string(cameraIndex));
        // We can either log or just return zeros
         spdlog::warn("[SystemCaptureFactory] Invalid camera index: {}", cameraIndex);
        return {0.0, 0};
    }
    double fps     = cameras_[cameraIndex]->getLastFPS();
    int    qSize   = cameras_[cameraIndex]->getQueueSize();

    // to debug output metric
   // std::cout << "Get Camera Metrics: GetLastFPS:"<< cameras_[cameraIndex]->getLastFPS() << " getAverageProcTime:"<< algo_->getAverageProcTime() << std::endl;
   // return {cameras_[cameraIndex]->getLastFPS(), cameras_[cameraIndex]->getQueueSize()};

    spdlog::debug("[SystemCaptureFactory] getCameraMetrics index={}, fps={}, queue={}", cameraIndex, fps, qSize);
    return {fps, qSize};
}

// ===========================
// Algorithm Metrics
// ===========================
inline std::tuple<double, double> SystemCaptureFactory::getAlgorithmMetrics() const {
    if (!algo_) {
        return {0.0, 0.0};
    }

    double fps = algo_->getFps();
    double pt  = algo_->getAverageProcTime();

    // to debug output metric
    //  std::cout << "Get Algorithm Metrics: GetFPS:"<< algo_->getFps() << " getAverageProc:"<< algo_->getAverageProcTime() << std::endl;
    // return {algo_->getFps(), algo_->getAverageProcTime()};

    spdlog::debug("[SystemCaptureFactory] getAlgorithmMetrics fps={}, procTime={}", fps, pt);
    return {fps, pt};
}

// ===========================
// System Control
// ===========================
inline void SystemCaptureFactory::initializeCapture(
    const CameraConfig& cameraConfig,
    const AlgorithmConfig& algoConfig)
{
    try {
        spdlog::info("[SystemCaptureFactory] Initializing Capture System...");
        cameraConfig_ = cameraConfig;

        // Adjust these if you want to dynamically set from camera config
        const int capWidth  = 320;//640;
        const int capHeight = 240;//480;
        std::vector<uint8_t> rgbOriginal(capWidth * capHeight * 3, 0);

        cameraConfig_ = cameraConfig;  // <--- Store for future usage
        // now cameraConfig_ is available to the entire class
        spdlog::info("[SystemCaptureFactory] Initializing Capture System...");

        // 1) Open & configure each camera
        for (size_t i = 0; i < cameras_.size(); ++i) {
            auto devicePath = "/dev/video" + std::to_string(i);
            spdlog::info("[SystemCaptureFactory] Opening camera at {}", devicePath);

            if (!cameras_[i]->openDevice(devicePath)) {
                reportError("Failed to open " + devicePath);
                continue;
            }
            if (!cameras_[i]->configure(cameraConfig_)) {
                reportError("Failed to configure camera " + devicePath);
            }
        }

        // 2) Configure the algorithm (if available)
        if (algo_ && !algo_->configure(algoConfig)) {
            spdlog::warn("[SystemCaptureFactory] Algorithm config issues may occur.");
        }

        // 3) Initialize SoC (optional)
        if (soc_) {
            soc_->initializeSoC();
        }
    
        // 4) If we have a display, configure it (optionally you can use DisplayConfig for advanced settings)
        if (display_) {
            configureDisplay(DisplayConfig{});
            // Attempt to init the SDL window
            display_->initializeDisplay(displayWidth_, displayHeight_);
        }

        // 5) Start threads
        running_ = true;
        
        // Start camera threads
        for (size_t i = 0; i < cameras_.size(); ++i) {
            auto devicePath = "/dev/video" + std::to_string(i);

            // start streaming
            if (!cameras_[i]->startStreaming()) {
                if (errorCallback_) {
                    errorCallback_("[SystemCaptureFactory] Failed to start streaming camera " + devicePath);
                }
                continue;
                spdlog::info("[SystemCaptureFactory] OK to start streaming camera " + devicePath);
            }
            cameraThreads_.emplace_back(&SystemCaptureFactory::captureLoop, this, cameras_[i].get());
        }

        // Algorithm thread
        if (algo_) {
            algorithmThread_ = std::thread(&SystemCaptureFactory::algorithmLoop, this, algo_.get());
        }

        // SoC monitoring
        if (soc_) {
            socThread_ = std::thread(&SystemCaptureFactory::SoCLoop, this, soc_.get());
        }

        // Display
        displayThread_ = std::thread(&SystemCaptureFactory::displayLoop, this);

        spdlog::info("[SystemCaptureFactory] Capture system initialized with {} camera(s).", cameras_.size());
    }
    catch (const std::exception& e) {
        spdlog::error("[SystemCaptureFactory] Initialization error: {}", e.what());
        stopCapture();
        throw;
    }
}

// ===========================
// Stop Capture !!!
// ===========================
inline void SystemCaptureFactory::stopCapture() {
    if (!running_) {
        return;
    }
    spdlog::info("[SystemCaptureFactory] Stopping system...");
    running_ = false;

    // Join camera threads
    for (auto& t : cameraThreads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    cameraThreads_.clear();
    // Join algorithm thread
    if (algorithmThread_.joinable()) {
        algorithmThread_.join();
    }
    // Join SoC thread
    if (socThread_.joinable()) {
        socThread_.join();
    }
    // Join display thread
    if (displayThread_.joinable()) {
        displayThread_.join();
    }
    // Free resources
    resetDevices();
    spdlog::info("[SystemCaptureFactory] Capture system stopped.");
}

inline void SystemCaptureFactory::resetDevices() {
    // stop/close each camera
    for (auto& camera : cameras_) {
        auto dataPtr = dynamic_cast<DataConcrete*>(camera.get());
        if (dataPtr) {
            dataPtr->resetDevice(); // close fd, unmap buffers, etc.
        }
    }
    cameras_.clear();
    // Clear algo, SoC, display
    algo_.reset();
    soc_.reset();
    display_.reset();

    spdlog::info("[SystemCaptureFactory] All devices have been reset.");
}

inline void SystemCaptureFactory::pauseAll() {
    paused_ = true;
    spdlog::info("[SystemCaptureFactory] System paused.");
}

inline void SystemCaptureFactory::resumeAll() {
    paused_ = false;
    spdlog::info("[SystemCaptureFactory] System resumed.");
}


inline void SystemCaptureFactory::setDisplay(std::shared_ptr<IDisplay> display) {
        display_ = std::move(display);
    }

// ===========================
// Configuration
// ===========================
inline void SystemCaptureFactory::configureCamera(size_t index, const CameraConfig& cfg) {
    if (index >= cameras_.size()) {
        reportError("Invalid camera index: " + std::to_string(index));
        return;
    }
    if (!cameras_[index]->configure(cfg)) {
        reportError("Failed to configure camera index " + std::to_string(index));
    }
    spdlog::info("[SystemCaptureFactory] Camera {} re-configured.", index);
}

inline void SystemCaptureFactory::configureAlgorithm(const AlgorithmConfig& cfg) {
    if (algo_ && !algo_->configure(cfg)) {
        spdlog::warn("[SystemCaptureFactory] Algorithm reconfiguration may have issues.");
    }
    spdlog::info("[SystemCaptureFactory] Algorithm re-configured.");
}

inline void SystemCaptureFactory::configureSoC(const SoCConfig& cfg) {
    if (soc_ && !soc_->configure(cfg)) {
        spdlog::warn("[SystemCaptureFactory] SoC reconfiguration may have issues.");
    }
    spdlog::info("[SystemCaptureFactory] SoC re-configured.");
}

inline void SystemCaptureFactory::configureDisplay(const DisplayConfig& cfg) {
    // If you have advanced display config, pass them to the display
    // For now just log; if your SdlDisplayConcrete has advanced config, pass it along
    if (display_) {
        display_->configure(cfg);
    }
    spdlog::info("[SystemCaptureFactory] Display re-configured.");
}

// ===========================
// Error Handling
// ===========================
inline void SystemCaptureFactory::setGlobalErrorCallback(std::function<void(const std::string&)> callback) {
    errorCallback_ = std::move(callback);
}

inline void SystemCaptureFactory::reportError(const std::string& msg) {
    if (errorCallback_) {
        errorCallback_(msg);
    } else {
        spdlog::error("[SystemCaptureFactory] {}", msg);
    }
}

// ===========================
// Thread Routines
// ===========================

/// captureLoop: Producer of frames from camera

/*
Camera → Algorithm: The captureLoop is a producer that dequeues frames from the camera driver (dequeFrame) and pushes them to frameQueue_. 
The algorithmLoop is a consumer that pops frames from frameQueue_ and runs processFrame()
*/
inline void SystemCaptureFactory::captureLoop(IData* camera) {
    while (running_) {
        if (paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        FrameData frame;
        // Attempt to dequeue a frame
        if (!camera->dequeFrame(frame.data, frame.size)) {
            spdlog::warn("[SystemCaptureFactory] WARNING: Failed to dequeue frame (no data?).");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        
        // // Convert from YUYV → RGB for display
        // if (display_) {
        //     if (frame.data != nullptr && frame.size > 0) {
        //             yuyvToRGB(static_cast<uint8_t*>(frame.data), rgbBuffer_.data(), cameraConfig_.width, cameraConfig_.height);
        //         } else {
        //             spdlog::warn("[SystemCaptureFactory] WARNING: Received invalid frame data!");
        //         }
        //          if (!rgbBuffer_.empty()) {
        //             // Show the original frame on display
        //             display_->updateOriginalFrame(rgbBuffer_.data(), cameraConfig_.width, cameraConfig_.height);
        //             std::cout << "[Display] Frame displayed successfully!\n";
        //         } else {
        //             spdlog::warn("[Display] WARNING: rgbBuffer_ is empty, no data to display.");
        //     }

        // Convert from YUYV → RGB for display
        if (display_) {
            if (frame.data != nullptr && frame.size > 0) {
                if (rgbBuffer_.empty()) {
                    // allocate once (width * height * 3)
                    rgbBuffer_.resize(cameraConfig_.width * cameraConfig_.height * 3, 0);
                }
                yuyvToRGB(static_cast<uint8_t*>(frame.data),
                          rgbBuffer_.data(),
                          cameraConfig_.width,
                          cameraConfig_.height);

                // show the original frame
                display_->updateOriginalFrame(rgbBuffer_.data(),
                                              cameraConfig_.width,
                                              cameraConfig_.height);
            }
        }

        // Next push frame into queue for the algorithm
        if (frameQueue_.size() < MAX_QUEUE_SIZE) {
            // e.g., store a timestamp
            frame.timestamp = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
            frameQueue_.push(frame);

            // Requeue the camera buffer
            camera->queueBuffer();
        } else {
            spdlog::warn("[SystemCaptureFactory] Dropping frame; queue is full (size={}).", MAX_QUEUE_SIZE);
        }
    }
    spdlog::info("[SystemCaptureFactory] Capture loop ended for one camera.");
}

// SoCLoop: SoC metrics
inline void SystemCaptureFactory::SoCLoop(ISoC* soc) {
    while (running_) {
        if (paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        auto perf = soc->getPerformance();
        spdlog::info("[SoC] CPU1 {}%@{}MHz, CPU: {}C, GPU: {}C",
                     perf.CPU1_Utilization_Percent, perf.CPU1_Frequency_MHz,
                     perf.CPU_Temperature_C, perf.GPU_Temperature_C);

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    spdlog::info("[SystemCaptureFactory] SoC monitoring loop ended.");
}

// // algorithmLoop: Consumer of frames
// inline void SystemCaptureFactory::algorithmLoop(IAlgorithm* alg) {
//     // // local buffer for “processed” frame
//     // const int capWidth  = 320;//640;
//     // const int capHeight = 240;//480;
//     // std::vector<uint8_t> localProcessed(capWidth * capHeight * 3, 0);

//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//             continue;
//         }
//         FrameData frame;
//         if (frameQueue_.pop(frame)) {
//             // process the frame
//             bool success = alg->processFrame(frame);
//             if (!success) {
//                 spdlog::warn("[SystemCaptureFactory] Algorithm failed to process frame.");
//             } else {
//                 // If the algorithm has a processed buffer, show it
//                 std::cout << "[Algorithm] Frame processed successfully!\n";
//                 // {
//                 //     std::lock_guard<std::mutex> lock(processedMutex_);
//                 //     processedRGB_.assign(localProcessed.begin(), localProcessed.end());
//                 // }

//                 // we update the "processed" display side
//                 // if (display_) {
//                 //     // e.g., algo->getProcessedBuffer() returns a pointer to the result
//                 //     auto* processedPtr = alg->getProcessedBuffer();
//                 //     int w = cameraConfig_.width;
//                 //     int h = cameraConfig_.height;
//                 //     display_->updateProcessedFrame(processedPtr, w, h);
//                 // }

//                 if (display_) {
//                 auto* procPtr = alg->getProcessedBuffer();
//                 if (procPtr) {
//                     // e.g. same camera dimension:
//                     display_->updateProcessedFrame(procPtr,
//                                                    cameraConfig_.width,
//                                                    cameraConfig_.height);
//                 }
//             }
            
//         } else {
//             std::this_thread::sleep_for(std::chrono::milliseconds(5));
//         }
//     }
//     spdlog::info("[SystemCaptureFactory] Algorithm loop ended.");
// }

/// algorithmLoop: Consumer of frames
inline void SystemCaptureFactory::algorithmLoop(IAlgorithm* alg) {
    while (running_) {
        if (paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        FrameData frame;
        if (!frameQueue_.pop(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        // process the frame
        bool success = alg->processFrame(frame);
        if (!success) {
            spdlog::warn("[SystemCaptureFactory] Algorithm failed to process frame.");
        } else {
            // If the algorithm has a processed buffer, show it
            if (display_) {
                auto* procPtr = alg->getProcessedBuffer();
                if (procPtr) {
                    // e.g. same camera dimension:
                    display_->updateProcessedFrame(procPtr,
                                                   cameraConfig_.width,
                                                   cameraConfig_.height);
                }
            }
        }
    }
    spdlog::info("[SystemCaptureFactory] Algorithm loop ended.");
}

/// displayLoop
// Within the SystemCaptureFactory class
 inline void SystemCaptureFactory::displayLoop() {
    while (running_) {
        if (paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        if (display_) {
            display_->renderAndPollEvents();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    spdlog::info("[SystemCaptureFactory] Display loop ended.");
}

// ===========================
// Capture Metrics methods
// ===========================

// ---- Capture Metrics methods ----
    std::shared_ptr<ISoC> SystemCaptureFactory::getSoC() const { 
            return soc_; 
        }

    std::shared_ptr<IData> SystemCaptureFactory::getCamera(int index) const { 
            // Handle potential out-of-bounds checks for camera index
            if (index >= 0 && index < cameras_.size()) { 
                return cameras_[index];  
            } else {
                // Handle the case where the requested camera index is invalid
                // For example, return nullptr or throw an exception
                return nullptr; 
            }
        }

    std::shared_ptr<IAlgorithm> SystemCaptureFactory::getAlgorithm() const { 
        return algo_; 
    }




//========================================================================================
// #pragma once

// #include <atomic>
// #include <thread>
// #include <vector>
// #include <functional>
// #include <mutex>
// #include <chrono>
// #include <tuple>
// #include <spdlog/spdlog.h>
// #include <SDL2/SDL.h>

// #include "../Interfaces/IData.h"
// #include "../Interfaces/IAlgorithm.h"
// #include "../Interfaces/ISoC.h"
// #include "../Interfaces/ISystemProfiling.h"

// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/AlgorithmConfig.h"
// #include "../SharedStructures/SoCConfig.h"
// #include "../SharedStructures/DisplayConfig.h"
// #include "../SharedStructures/SharedQueue.h"
// #include "../SharedStructures/jetsonNanoInfo.h"

// #include "../Concretes/SdlDisplayConcrete.h"
// #include "../Concretes/DataConcrete.h"
// #include "SystemModellingFactory.h"

// /**
//  * @class SystemCaptureFactory
//  * @brief Coordinates the main “capture system” in an asynchronous pipeline:
//  *  - Multiple cameras (async streaming)
//  *  - A shared frame queue (producer-consumer)
//  *  - A separate algorithm thread pulling from the queue
//  *  - A display thread (optional)
//  *  - A SoC monitoring thread
//  *
//  * Implements ISystemProfiling to provide SoC, camera, and algorithm metrics 
//  * (queried by e.g. SystemProfilingFactory).
//  *
//  * Usage:
//  *  1. Construct with pointers to SoC, Algorithm, and camera(s).
//  *  2. Call initializeCapture(cameraConfig, algoConfig) to open devices, configure modules, and start threads.
//  *  3. Each camera pushes frames into frameQueue_ (captureLoop).
//  *  4. The algorithm thread pops frames (algorithmLoop).
//  *  5. SoC monitoring logs SoC stats (SoCLoop).
//  *  6. Optionally, a display thread (displayLoop).
//  */
// class SystemCaptureFactory : public ISystemProfiling {
// public:
//     // Constructors
//     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                          std::shared_ptr<IAlgorithm> algo,
//                          std::shared_ptr<IData> camera0,
//                          std::shared_ptr<IData> camera1);

//     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                          std::shared_ptr<IAlgorithm> algo,
//                          std::shared_ptr<IData> camera);

//     ~SystemCaptureFactory() override;

//     // ISystemProfiling interface

//      // Get SoC metrics: CPU usage, GPU usage, CPU temp, GPU temp
//     JetsonNanoInfo getSoCMetrics() const override;
//     // Get camera metrics: FPS and queue size for a specific camera
//     std::tuple<double, int> getCameraMetrics(int cameraIndex) override;
//     // Get algorithm metrics: FPS and average processing time
//     std::tuple<double, double> getAlgorithmMetrics() const override;



//     // System control
//     void initializeCapture(const CameraConfig& cameraConfig, const AlgorithmConfig& algoConfig);
//     void stopCapture();
//     void resetDevices();
//     void pauseAll();
//     void resumeAll();

//     // Configuration
//     void configureCamera(size_t index, const CameraConfig& cfg);
//     void configureAlgorithm(const AlgorithmConfig& cfg);
//     void configureSoC(const SoCConfig& cfg);
//     void configureDisplay(const DisplayConfig& cfg);

//     // Error handling
//     void setGlobalErrorCallback(std::function<void(const std::string&)>) ;

// private:
//     // Threads
//     void captureLoop(IData* camera);     // Producer loop for each camera
//     void SoCLoop(ISoC* soc);         // SoC monitoring
//     void algorithmLoop(IAlgorithm* alg); // Consumer loop
//     void displayLoop();                  // Optional UI thread

//     // Utility
//     void reportError(const std::string& msg);

// private:
//     std::atomic<bool> running_;
//     std::atomic<bool> paused_;

//     std::shared_ptr<ISoC> soc_;
//     std::shared_ptr<IAlgorithm> algo_;
//     std::vector<std::shared_ptr<IData>> cameras_;
//     std::shared_ptr<IDisplay> display_;

//     // Threads for each component (camera, SoC, display, algorithm)
//     std::vector<std::thread> cameraThreads_;
//     std::thread socThread_;
//     std::thread displayThread_;
//     std::thread algorithmThread_;

//     SharedQueue<FrameData> frameQueue_; ///< Producer-consumer queue for frames

//     std::function<void(const std::string&)> errorCallback_;

//     // Optional display resolution
//     int displayWidth_  = 320;
//     int displayHeight_ = 240;

//     // Example max queue size to avoid unbounded growth
//     static constexpr size_t MAX_QUEUE_SIZE = 10;

//     std::mutex processedMutex_; 
//     std::vector<uint8_t> processedRGB_;
// };

// // ================================================
// // Implementation (can be moved to .cpp if desired)
// // ================================================

// // Constructor with 2 cameras   
// inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                                                   std::shared_ptr<IAlgorithm> algo,
//                                                   std::shared_ptr<IData> camera0,
//                                                   std::shared_ptr<IData> camera1)
//     : soc_(std::move(soc))
//     , algo_(std::move(algo))
//     , running_(false)
//     , paused_(false)
//     , errorCallback_(nullptr)
// {
//     if (!camera0 || !camera1) {
//         throw std::invalid_argument("[SystemCaptureFactory] Null camera pointer(s).");
//     }
//     cameras_.push_back(std::move(camera0));
//     cameras_.push_back(std::move(camera1));

//     if (cameras_.empty()) {
//         throw std::invalid_argument("[SystemCaptureFactory] At least one camera is required.");
//     }
// }

// // Constructor with 1 camera
// inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                                                   std::shared_ptr<IAlgorithm> algo,
//                                                   std::shared_ptr<IData> camera)
//     : soc_(std::move(soc))
//     , algo_(std::move(algo))
//     , running_(false)
//     , paused_(false)
//     , errorCallback_(nullptr)
// {
//     // push the single camera into the vector
//     cameras_.push_back(std::move(camera));

//     if (cameras_.empty()) {
//         throw std::invalid_argument("[SystemCaptureFactory] At least one camera is required.");
//     }
// }

// // Destructor
// inline SystemCaptureFactory::~SystemCaptureFactory() {
//     stopCapture();
// }

// // ISystemProfiling
// inline JetsonNanoInfo SystemCaptureFactory::getSoCMetrics() const {
//     return soc_ ? soc_->getPerformance() : JetsonNanoInfo{};
// }

// inline std::tuple<double, int> SystemCaptureFactory::getCameraMetrics(int cameraIndex) {
//     if (cameraIndex < 0 || static_cast<size_t>(cameraIndex) >= cameras_.size()) {
//         reportError("Invalid camera index: " + std::to_string(cameraIndex));
//         return {0.0, 0};
//     }
//     return {cameras_[cameraIndex]->getLastFPS(), cameras_[cameraIndex]->getQueueSize()};
// }

// inline std::tuple<double, double> SystemCaptureFactory::getAlgorithmMetrics() const {
//     // e.g., fps + average processing time
//     return algo_ 
//         ? std::make_tuple(algo_->getFps(), algo_->getAverageProcTime()) 
//         : std::make_tuple(0.0, 0.0);
// }

// // =========================
// // System Control
// // =========================

//  inline void SystemCaptureFactory::initializeCapture(
//     const CameraConfig& cameraConfig, 
//     const AlgorithmConfig& algoConfig) 
// {
//     try {
//         spdlog::info("[SystemCaptureFactory] Initializing Capture System...");
        
//         // STEP 1: Open and configure cameras before starting threads
//         for (size_t i = 0; i < cameras_.size(); ++i) {
//             auto devicePath = "/dev/video" + std::to_string(i);

//             if (!cameras_[i]->openDevice(devicePath)) {
//                 reportError("[SystemCaptureFactory] Failed to open " + devicePath);
//                 continue; // Skip to next camera
//             }

//             if (!cameras_[i]->configure(cameraConfig)) {
//                 reportError("[SystemCaptureFactory] Failed to configure camera at " + devicePath);
//                 continue; // Skip to next camera
//             }
//         }

//         // STEP 2: Configure algorithm (if available)
//         if (algo_ && !algo_->configure(algoConfig)) {
//             spdlog::warn("[SystemCaptureFactory] Algorithm configuration issues may occur.");
//         }

//         // STEP 3: Initialize SoC monitoring
//         if (soc_) {
//             soc_->initializeSoC();
//         }

//         // STEP 4: Initialize SDL Display
//         if (display_) {
//             display_->initializeDisplay(displayWidth_, displayHeight_);
//         }

//         // STEP 5: Start streaming cameras and spawn threads
//         running_ = true;
//         for (size_t i = 0; i < cameras_.size(); ++i) {
//             auto devicePath = "/dev/video" + std::to_string(i);

//             if (!cameras_[i]->startStreaming()) {
//                 errorCallback_("[SystemCaptureFactory] Failed to start streaming camera " + devicePath);
//                 continue; // Skip to next camera
//             }

//             // Start the camera thread **only if streaming was successful**
//             cameraThreads_.emplace_back(&SystemCaptureFactory::captureLoop, this, cameras_[i].get());
//         }

//         // STEP 6: Start Algorithm, SoC Monitoring, and Display Threads
//         if (algo_) {
//             algorithmThread_ = std::thread(&SystemCaptureFactory::algorithmLoop, this, algo_.get());
//         }

//         if (soc_) {
//             socThread_ = std::thread(&SystemCaptureFactory::SoCLoop, this, soc_.get());
//         }

//         displayThread_ = std::thread(&SystemCaptureFactory::displayLoop, this);

//         spdlog::info("[SystemCaptureFactory] Capture system initialized with {} cameras.", cameras_.size());
//     }
//     catch (const std::exception& e) {
//         spdlog::error("[SystemCaptureFactory] Initialization error: {}", e.what());
//         stopCapture();
//         throw;
//     }
// }


// inline void SystemCaptureFactory::stopCapture() {
//     if (!running_) return;
//       spdlog::info("[SystemCaptureFactory] Stopping system...");
//     running_ = false;

//     // Join camera threads
//     // Stop all threads
//     for (auto& t : cameraThreads_) {
//         if (t.joinable()) {
//             t.join();
//         }
//     }
//     cameraThreads_.clear();

//     // Join algorithm thread
//     if (algorithmThread_.joinable()) {
//         algorithmThread_.join();
//     }

//     // Join SoC thread
//     if (socThread_.joinable()) {
//         socThread_.join();
//     }

//     // Join display thread
//     if (displayThread_.joinable()) {
//         displayThread_.join();
//     }
//      // Reset devices and free resources
//     resetDevices();

//     spdlog::info("[SystemCaptureFactory] Capture system stopped.");
// }

// inline void SystemCaptureFactory::pauseAll() {
//     paused_ = true;
//     spdlog::info("[SystemCaptureFactory] System paused.");
// }

// inline void SystemCaptureFactory::resumeAll() {
//     paused_ = false;
//     spdlog::info("[SystemCaptureFactory] System resumed.");
// }

// void SystemCaptureFactory::resetDevices() {
//     for (auto& camera : cameras_) {
//         if (camera) {
//             dynamic_cast<DataConcrete*>(camera.get())->resetDevice();
//         }
//     }
//     cameras_.clear();
//     algo_.reset();
//     soc_.reset();
//     display_.reset();
//     spdlog::info("[SystemCaptureFactory] All devices have been reset.");
// }



// // =========================
// // Configuration
// // =========================
// inline void SystemCaptureFactory::configureCamera(size_t index, const CameraConfig& cfg) {
//     if (index >= cameras_.size()) {
//         reportError("Invalid camera index: " + std::to_string(index));
//         return;
//     }
//     if (!cameras_[index]->configure(cfg)) {
//         reportError("Failed to configure camera index " + std::to_string(index));
//     }
//     spdlog::info("[SystemCaptureFactory] Camera {} re-configured.", index);
// }

// inline void SystemCaptureFactory::configureAlgorithm(const AlgorithmConfig& cfg) {
//     if (algo_ && !algo_->configure(cfg)) {
//         spdlog::warn("[SystemCaptureFactory] Algorithm reconfiguration may have issues.");
//     }
//     spdlog::info("[SystemCaptureFactory] Algorithm re-configured.");
// }

// inline void SystemCaptureFactory::configureSoC(const SoCConfig& cfg) {
//     if (soc_ && !soc_->configure(cfg)) {
//         spdlog::warn("[SystemCaptureFactory] SoC reconfiguration may have issues.");
//     }
//     spdlog::info("[SystemCaptureFactory] SoC re-configured.");
// }

// inline void SystemCaptureFactory::configureDisplay(const DisplayConfig& cfg) {
//     if (display_) {
//         display_->configure(cfg);
//     }
//     spdlog::info("[SystemCaptureFactory] Display re-configured.");
// }

// // =========================
// // Error Handling
// // =========================
// inline void SystemCaptureFactory::setGlobalErrorCallback(std::function<void(const std::string&)> callback) {
//     errorCallback_ = std::move(callback);
// }

// inline void SystemCaptureFactory::reportError(const std::string& msg) {
//     if (errorCallback_) {
//         errorCallback_(msg);
//     } else {
//         spdlog::error("[SystemCaptureFactory] {}", msg);
//     }
// }

// // =========================
// // Thread Routines
// // =========================

// // Capture Loop (Producer)
// inline void SystemCaptureFactory::captureLoop(IData* camera) {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//             continue;
//         }

//         FrameData frame;
//         // Attempt to dequeue a frame from the camera
//         if (camera->dequeFrame(frame.data, frame.size)) {
//             // Optional queue size limit
//             if (frameQueue_.size() < MAX_QUEUE_SIZE) {
//                 frameQueue_.push(frame);
//             } else {
//                 spdlog::warn("Frame dropped due to a full queue (size={}).", MAX_QUEUE_SIZE);
//             }
//         } else {
//             // If dequeFrame fails or no frame, wait briefly to avoid busy loop
//             std::this_thread::sleep_for(std::chrono::milliseconds(5));
//         }
//     }
//     spdlog::info("[SystemCaptureFactory] Capture loop ended for one camera.");
// }

// // Monitor Loop (SoC)
// inline void SystemCaptureFactory::SoCLoop(ISoC* soc) {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(500));
//             continue;
//         }
//         // Obtain SoC performance
//         auto perf = soc->getPerformance();
//         spdlog::info("[SoC] CPU1 {}%@{}MHz, CPU: {}C, GPU: {}C",
//                      perf.CPU1_Utilization_Percent, perf.CPU1_Frequency_MHz,
//                      perf.CPU_Temperature_C, perf.GPU_Temperature_C);

//         std::this_thread::sleep_for(std::chrono::seconds(1));
//     }
//     spdlog::info("[SystemCaptureFactory] SoC monitoring loop ended.");
// }

// // Algorithm Loop (Consumer)
// inline void SystemCaptureFactory::algorithmLoop(IAlgorithm* alg) {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//             continue;
//         }

//         FrameData frame;
//         if (frameQueue_.pop(frame)) {
//             // processFrame is typically synchronous; if it's time-consuming, consider 
//             // a separate thread or concurrency approach inside the algorithm.
//             alg->processFrame(frame);
//         } else {
//             // No frame available; sleep to avoid spinning
//             std::this_thread::sleep_for(std::chrono::milliseconds(5));
//         }
//     }
//     spdlog::info("[SystemCaptureFactory] Algorithm loop ended.");
// }

// // Display Loop
// inline void SystemCaptureFactory::displayLoop() {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(50));
//             continue;
//         }
//         // If you have a display, call its render method
//         if (display_) {
//             display_->renderAndPollEvents();
//         } else {
//             // If no display, just idle
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//         }
//     }
//     spdlog::info("[SystemCaptureFactory] Display loop ended.");
// }


