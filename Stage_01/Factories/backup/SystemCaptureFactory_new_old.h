// SystemCaptureFactory_new.h

#ifndef SYSTEMCAPTUREFACTORY_NEW_H
#define SYSTEMCAPTUREFACTORY_NEW_H

#include <atomic>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <spdlog/spdlog.h>
#include <SDL2/SDL.h>
#include <fmt/format.h>

// Interfaces
#include "../Interfaces/IData.h"
#include "../Interfaces/IAlgorithm.h"
#include "../Interfaces/ISoC.h"
#include "../Interfaces/IDisplay.h"
#include "../Interfaces/ISystemProfiling.h"

// Shared Structures
#include "../SharedStructures/FrameData.h"
#include "../SharedStructures/CameraConfig.h"
#include "../SharedStructures/AlgorithmConfig.h"
#include "../SharedStructures/SoCConfig.h"
#include "../SharedStructures/DisplayConfig.h"
#include "../SharedStructures/SharedQueue.h"

// Thread Management
#include "../SharedStructures/ThreadManager.h"

// Concrete Implementations
#include "../Concretes/DataConcrete_new.h"
#include "../Concretes/AlgorithmConcrete_new.h"
#include "../Concretes/SdlDisplayConcrete_new.h"
#include "../Concretes/SoCConcrete.h"

/**
 * @struct CaptureSelection
 * @brief Determines which system components to enable.
 */
struct CaptureSelection {
    bool enableCamera   = true;  // Enable single-camera capture?
    bool enableAlgorithm = true; // Enable algorithm processing?
    bool enableSoC       = false; // Enable SoC monitoring?
};

/**
 * @class SystemCaptureFactory
 * @brief Manages the end-to-end capture, processing, and display pipeline.
 *
 * **Pipeline Workflow:**
 * 1️⃣ Camera captures frames and pushes them to `displayOrigQueue_`.
 * 2️⃣ Algorithm reads frames from `algoQueue_` and pushes processed frames to `processedQueue_`.
 * 3️⃣ Display retrieves frames from `displayOrigQueue_` and `processedQueue_` for rendering.
 */
class SystemCaptureFactory : public ISystemProfiling {
public:
    // Constructor for single-camera setup
    SystemCaptureFactory(std::shared_ptr<ISoC> soc,
                         std::shared_ptr<IAlgorithm> algo,
                         std::shared_ptr<IData> camera);
    ~SystemCaptureFactory() override;

    // System lifecycle control
    void initializeCapture(const CameraConfig& cameraConfig, 
                           const AlgorithmConfig& algoConfig, 
                           const CaptureSelection& captureSelection);
    void stopCapture();
    void pauseAll();
    void resumeAll();

    // Display management
    void setDisplay(std::shared_ptr<IDisplay> display);
    std::shared_ptr<IDisplay> getDisplay() const;

    // Configuration
    void configureCamera(const CameraConfig& cfg);
    void configureAlgorithm(const AlgorithmConfig& cfg);
    void configureSoC(const SoCConfig& cfg);
    void configureDisplay(const DisplayConfig& cfg);

    // Metrics retrieval
    JetsonNanoInfo getSoCMetrics() const override;
   // std::tuple<double, int> getCameraMetrics() const override;
    std::tuple<double, double> getAlgorithmMetrics() const override;

    // Get system components
    std::shared_ptr<ISoC> getSoC() const;
    std::shared_ptr<IData> getCamera() const;
    std::shared_ptr<IAlgorithm> getAlgorithm() const;

    // Get camera metrics
     std::tuple<double, int> getCameraMetrics(int cameraIndex) const override {
        if (!camera_) {
            spdlog::warn("[SystemCaptureFactory] No camera instance available.");
            return {0.0, 0};
        }
        return {camera_->getLastFPS(), camera_->getQueueSize()};
    }

private:
    // Core system loops
    void captureLoop(IData* camera);
    void algorithmLoop();
    void SoCLoop(ISoC* soc);

    // Error handling
    void reportError(const std::string& msg);

private:
    std::atomic<bool> running_;
    std::atomic<bool> paused_;

    std::shared_ptr<ISoC> soc_;
    std::shared_ptr<IAlgorithm> algo_;
    std::shared_ptr<IData> camera_;
    std::shared_ptr<IDisplay> display_;

    ThreadManager threadManager_;

    std::shared_ptr<SharedQueue<FrameData>> algoQueue_;
    std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue_;
    std::shared_ptr<SharedQueue<FrameData>> processedQueue_;
};

// ===================================================
// == 🔹 Implementation ==
// ===================================================

inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
                                                  std::shared_ptr<IAlgorithm> algo,
                                                  std::shared_ptr<IData> camera)
    : running_(false), paused_(false), soc_(std::move(soc)), algo_(std::move(algo)), camera_(std::move(camera)) {
    if (!camera_) {
        throw std::invalid_argument("[SystemCaptureFactory] Camera pointer is null!");
    }
}

inline SystemCaptureFactory::~SystemCaptureFactory() {
    stopCapture();
}

inline void SystemCaptureFactory::initializeCapture(const CameraConfig& cameraConfig,
                                                    const AlgorithmConfig& algoConfig,
                                                    const CaptureSelection& selection) {
    try {
        spdlog::info("[SystemCaptureFactory] Initializing capture system...");

        if (!camera_->openDevice("/dev/video0")) {
            throw std::runtime_error("[SystemCaptureFactory] Failed to open /dev/video0");
        }
        if (!camera_->configure(cameraConfig)) {
            throw std::runtime_error("[SystemCaptureFactory] Failed to configure camera");
        }

        if (selection.enableCamera) {
            camera_->startStreaming();
        }

        if (algo_ && selection.enableAlgorithm) {
            algo_->configure(algoConfig);
        }

        if (soc_ && selection.enableSoC) {
            soc_->initializeSoC();
        }

        if (display_) {
            DisplayConfig dispCfg = {cameraConfig.width, cameraConfig.height, false, "Framework Display"};
            display_->configure(dispCfg);
            display_->initializeDisplay(dispCfg.width, dispCfg.height);
        }

        running_ = true;
        paused_ = false;

        algoQueue_ = std::make_shared<SharedQueue<FrameData>>(10);
        displayOrigQueue_ = std::make_shared<SharedQueue<FrameData>>(10);
        processedQueue_ = std::make_shared<SharedQueue<FrameData>>(10);

        if (selection.enableSoC) {
            threadManager_.addThread("SoCMonitor", std::thread([this]() { SoCLoop(soc_.get()); }));
        }
        if (selection.enableCamera) {
            threadManager_.addThread("CameraCapture", std::thread([this]() { captureLoop(camera_.get()); }));
        }
        if (selection.enableAlgorithm) {
            threadManager_.addThread("AlgorithmProcessing", std::thread(&SystemCaptureFactory::algorithmLoop, this));
        }

        spdlog::info("[SystemCaptureFactory] Initialization complete.");
    } catch (const std::exception& e) {
        reportError(e.what());
        stopCapture();
        throw;
    }
}

inline void SystemCaptureFactory::stopCapture() {
    if (!running_) return;
    running_ = false;
    if (camera_) camera_->stopStreaming();
    threadManager_.joinAll();
    spdlog::info("[SystemCaptureFactory] Capture system stopped.");
}

inline void SystemCaptureFactory::captureLoop(IData* camera) {
    while (running_) {
        void* dataPtr = nullptr;
        size_t sizeBytes = 0;
        size_t bufferIndex = 0;

        if (!camera->dequeFrame(dataPtr, sizeBytes, bufferIndex)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        FrameData frame;
        frame.dataVec.assign(static_cast<uint8_t*>(dataPtr), static_cast<uint8_t*>(dataPtr) + sizeBytes);
        frame.size = sizeBytes;
        frame.width = 640;
        frame.height = 480;

        displayOrigQueue_->push(frame);
        camera->queueBuffer(bufferIndex);
    }
}

inline void SystemCaptureFactory::algorithmLoop() {
    while (running_) {
        FrameData frame;
        if (!algoQueue_->pop(frame)) continue;
        algo_->processFrame(frame);
    }
}

inline void SystemCaptureFactory::SoCLoop(ISoC* soc) {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

inline void SystemCaptureFactory::reportError(const std::string& msg) {
    spdlog::error("[SystemCaptureFactory] {}", msg);
}

void SystemCaptureFactory::setDisplay(std::shared_ptr<IDisplay> display) {
    //display_ = std::move(display);
    display_ = display;
}
std::shared_ptr<IDisplay> SystemCaptureFactory::getDisplay() const {
    return display_;
}
JetsonNanoInfo SystemCaptureFactory::getSoCMetrics() const {
    return soc_ ? soc_->getPerformance() : JetsonNanoInfo{};
}
std::tuple<double, double> SystemCaptureFactory::getAlgorithmMetrics() const {
    return {algo_ ? algo_->getFps() : 0.0, algo_ ? algo_->getAverageProcTime() : 0.0};
}


#endif // SYSTEMCAPTUREFACTORY_NEW_H



// //==========================================================
// #ifndef SYSTEMCAPTUREFACTORY_NEW_H
// #define SYSTEMCAPTUREFACTORY_NEW_H

// #include <atomic>
// #include <vector>
// #include <memory>
// #include <unordered_set>
// #include <chrono>
// #include <tuple>
// #include <condition_variable>
// #include <mutex>
// #include <stdexcept>
// #include <functional>
// #include <spdlog/spdlog.h>
// #include <SDL2/SDL.h>
// #include <fmt/format.h>

// // Include interfaces
// #include "../Interfaces/IData.h"
// #include "../Interfaces/IAlgorithm.h"
// #include "../Interfaces/ISoC.h"
// #include "../Interfaces/IDisplay.h"
// #include "../Interfaces/ISystemProfiling.h"

// // Include shared structures and SharedQueue
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/AlgorithmConfig.h"
// #include "../SharedStructures/SoCConfig.h"
// #include "../SharedStructures/DisplayConfig.h"
// #include "../SharedStructures/SharedQueue.h"

// // Include ThreadManager
// #include "../SharedStructures/ThreadManager.h"

// // Include concrete implementations (the _new versions)
// #include "../Concretes/DataConcrete_new.h"      // Updated: constructor takes shared queues.
// #include "../Concretes/AlgorithmConcrete_new.h"   // Updated: constructor takes shared queues and ThreadManager reference.
// #include "../Concretes/SdlDisplayConcrete_new.h"    // Updated: constructor takes shared queues.
// #include "../Concretes/SoCConcrete.h"


// /**
//  * // Forward declare your struct if needed
// // struct CaptureSelection { bool enableCameras = true; bool enableAlgorithm = true; bool enableSoC = false; };
//  * @brief A struct to indicate which pipeline components you want to enable.
//  */
// struct CaptureSelection {
//     bool enableCameras   = true;   // Start camera capture threads?
//     bool enableAlgorithm = true;   // Start algorithm thread?
//     bool enableSoC       = false;  // Start SoC monitoring thread?
//     // You can add more fields if needed (e.g., enableAudio, enableNetwork, etc.)
// };


// /**
//  * @class SystemCaptureFactory
//  * @brief Coordinates the capture system in an asynchronous pipeline.
//  *
//  * The pipeline works as follows:
//  *  - DataConcrete captures frames and pushes them into shared queues.
//  *  - AlgorithmConcrete reads frames from its input queue and pushes processed frames into another queue.
//  *  - SdlDisplayConcrete reads frames from both the original and processed queues and renders them.
//  *
//  * **Important:** In our design, all SDL calls (e.g. SDL_Init, rendering, and event polling)
//  * are performed on the main thread. Thus, the display initialization and the display loop
//  * are invoked from the main thread (for example, via a manual loop in main()), and no additional
//  * thread is spawned in the factory to handle display.
//  */
// class SystemCaptureFactory : public ISystemProfiling {
// public:
//     // Constructors: for a single camera.
//     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                          std::shared_ptr<IAlgorithm> algo,
//                          std::shared_ptr<IData> camera);
//     ~SystemCaptureFactory() override;

//     // ISystemProfiling interface methods
//     JetsonNanoInfo getSoCMetrics() const override;
//     std::tuple<double, int> getCameraMetrics(int cameraIndex) const override;
//     std::tuple<double, double> getAlgorithmMetrics() const override;

//     // System control methods
//     void initializeCapture(const CameraConfig& cameraConfig,
//                             const AlgorithmConfig& algoConfig,
//                             const CaptureSelection& captureSelection);

//     void stopCapture();
//     void resetDevices();
//     void pauseAll();
//     void resumeAll();

//     // Display management.
//     void setDisplay(std::shared_ptr<IDisplay> display);
//     std::shared_ptr<IDisplay> getDisplay() const;

//     // Configuration methods.
//     void configureCamera(size_t index, const CameraConfig& cfg);
//     void configureAlgorithm(const AlgorithmConfig& cfg);
//     void configureSoC(const SoCConfig& cfg);
//     void configureDisplay(const DisplayConfig& cfg);

//     // Error handling.
//     void setGlobalErrorCallback(std::function<void(const std::string&)>);
    
//     // Helper method to check if a component is enabled.
//     bool isComponentEnabled(const std::vector<Component>& components, Component c);

//     // Accessors for components.
//     std::shared_ptr<ISoC> getSoC() const;
//     std::shared_ptr<IData> getCamera(int index = 0) const;
//     std::shared_ptr<IAlgorithm> getAlgorithm() const;

// private:
//     // Thread routines.
//     void captureLoop(IData* camera);
//     void algorithmLoop();
//     void SoCLoop(ISoC* soc);
//     void reportError(const std::string& msg);

// private:
//     std::atomic<bool> running_;
//     std::atomic<bool> paused_;
//     std::shared_ptr<ISoC>      soc_;
//     std::shared_ptr<IAlgorithm> algo_;
//     std::shared_ptr<IData>      camera_;  // Only 1 camera

//    // std::vector<std::shared_ptr<IData>> cameras_;  // One or more cameras
//     std::shared_ptr<IDisplay>  display_;           // The display component

//     // Centralized Thread Management.
//     ThreadManager threadManager_;

//     // Global error callback.
//     std::function<void(const std::string&)> errorCallback_;

//     // Display helpers.
//     int displayWidth_  = 640;
//     int displayHeight_ = 480;

//     // Camera configuration.
//     CameraConfig cameraConfig_;

//     // Mutex for buffer tracking.
//     std::mutex bufferMutex_;
//     std::unordered_set<size_t> activeBuffers_;  // Tracks buffers in use

//     // Condition variable for thread control.
//     std::mutex mutex_;
//     std::condition_variable cv_;
//     bool processing_ = false;

//     // Internal shared queues (used by DataConcrete, AlgorithmConcrete, and SdlDisplayConcrete).
//     std::shared_ptr<SharedQueue<FrameData>> algoQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> processedQueue_;
// };

// // ===================== Implementation =====================

// // --- Constructors ---
// inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                                                     std::shared_ptr<IAlgorithm> algo,
//                                                     std::shared_ptr<IData> camera)
//     : running_(false), paused_(false), soc_(std::move(soc)), algo_(std::move(algo)),camera_(std::move(camera))
// {
//    if (!camera_) {
//         throw std::invalid_argument("Camera pointer is null!");
//     }
// }


// inline SystemCaptureFactory::~SystemCaptureFactory() {
//     stopCapture();
// }

// // --- ISystemProfiling Methods ---
// inline JetsonNanoInfo SystemCaptureFactory::getSoCMetrics() const {
//     return soc_ ? soc_->getPerformance() : JetsonNanoInfo{};
// }

// inline std::tuple<double, int> SystemCaptureFactory::getCameraMetrics(int cameraIndex) const {
//     if (cameraIndex < 0 || static_cast<size_t>(cameraIndex) >= cameras_.size()) {
//         spdlog::warn("[SystemCaptureFactory] Invalid camera index: {}", cameraIndex);
//         return {0.0, 0};
//     }
//     double fps = cameras_[cameraIndex]->getLastFPS();
//     int qSize  = cameras_[cameraIndex]->getQueueSize();
//     spdlog::debug("[SystemCaptureFactory] getCameraMetrics index={}, fps={}, queue={}",
//                   cameraIndex, fps, qSize);
//     return {fps, qSize};
// }

// inline std::tuple<double, double> SystemCaptureFactory::getAlgorithmMetrics() const {
//     if (!algo_) {
//         return {0.0, 0.0};
//     }
//     double fps = algo_->getFps();
//     double pt  = algo_->getAverageProcTime();
//     spdlog::debug("[SystemCaptureFactory] getAlgorithmMetrics fps={}, procTime={}", fps, pt);
//     return {fps, pt};
// }

// // --- System Control Methods ---
// /**
//  * @brief Initialize capture system with flexible component selection.
//  * 
//  * @param cameraConfig    Configuration for the camera(s).
//  * @param algoConfig      Configuration for the algorithm.
//  * @param componentsToStart  A list of components to start:
//  *                           e.g. { Component::Camera, Component::Algorithm }.
//  */
// inline void SystemCaptureFactory::initializeCapture(const CameraConfig& cameraConfig,
//                                                     const AlgorithmConfig& algoConfig,
//                                                     const CaptureSelection& captureSelection)
// {
//     try {
//         spdlog::info("[SystemCaptureFactory] Initializing capture system...");
//         cameraConfig_ = cameraConfig;

//         // 1) Configure Cameras (open device, etc.) 
//         //    (We usually do this regardless of enableCameras, unless you want to skip even opening the device.)
//         for (size_t i = 0; i < cameras_.size(); ++i) {
//             auto devicePath = fmt::format("/dev/video{}", i);
//             spdlog::info("[SystemCaptureFactory] Opening camera {} at {}", i, devicePath);

//             if (!cameras_[i]->openDevice(devicePath)) {
//                 reportError(fmt::format("Failed to open {}", devicePath));
//                 continue;
//             }
//             if (!cameras_[i]->configure(cameraConfig_)) {
//                 reportError(fmt::format("Failed to configure {}", devicePath));
//                 continue;
//             }
//         }

//         // 2) Start streaming on cameras, only if requested
//         if (captureSelection.enableCameras) {
//             spdlog::info("[SystemCaptureFactory] Starting streaming for all cameras...");
//             for (size_t i = 0; i < cameras_.size(); ++i) {
//                 if (!cameras_[i]->startStreaming()) {
//                     spdlog::warn("[SystemCaptureFactory] Failed to start streaming for camera {}", i);
//                 }
//             }
//         } else {
//             spdlog::info("[SystemCaptureFactory] Cameras not started (disabled by captureSelection).");
//         }

//         // 3) Configure the algorithm
//         if (algo_) {
//             if (!algo_->configure(algoConfig)) {
//                 reportError("Algorithm configuration failed.");
//             }
//         }

//         // 4) Initialize SoC (optional, only if you have an SoC pointer)
//         if (soc_) {
//             soc_->initializeSoC();
//         }

//         // 5) Initialize display if available (on the main thread, as usual)
//         if (display_) {
//             DisplayConfig dispCfg;
//             dispCfg.width       = cameraConfig.width;
//             dispCfg.height      = cameraConfig.height;
//             dispCfg.fullScreen  = false;
//             dispCfg.windowTitle = "Framework Display";

//             display_->configure(dispCfg);
//             display_->initializeDisplay(dispCfg.width, dispCfg.height);
//             spdlog::info("[SystemCaptureFactory] Opening Framework Display {}x{}",
//                          dispCfg.width, dispCfg.height);
//         } else {
//             spdlog::error("[SystemCaptureFactory] No display instance found!");
//         }

//         // 6) Mark system as running and create internal queues if needed
//         running_ = true;
//         paused_  = false;

//         if (!algoQueue_ || !displayOrigQueue_ || !processedQueue_) {
//             algoQueue_        = std::make_shared<SharedQueue<FrameData>>(10);
//             displayOrigQueue_ = std::make_shared<SharedQueue<FrameData>>(10);
//             processedQueue_   = std::make_shared<SharedQueue<FrameData>>(10);
//         }

//         // -------------------------------
//         // Using the ThreadManager
//         // -------------------------------

//         // (a) SoC monitoring thread, only if enabled
//         if (soc_ && captureSelection.enableSoC) {
//             spdlog::info("[SystemCaptureFactory] Spawning SoC monitoring thread...");
//             threadManager_.addThread(Component::Custom, 
//                 std::thread([this]() {
//                     this->SoCLoop(soc_.get());
//                 })
//             );
//         }

//         // (b) Camera capture threads, only if enabled
//         if (captureSelection.enableCameras) {
//             for (size_t i = 0; i < cameras_.size(); ++i) {
//                 spdlog::info("[SystemCaptureFactory] Spawning capture thread for camera {}", i);
//                 auto cameraKey = fmt::format("Camera{}", i);
//                 threadManager_.addThread(cameraKey, 
//                     std::thread([this, i]() {
//                         this->captureLoop(cameras_[i].get());
//                     })
//                 );
//             }
//         }

//         // (c) Algorithm thread, only if enabled
//         if (captureSelection.enableAlgorithm) {
//             spdlog::info("[SystemCaptureFactory] Spawning algorithm thread...");
//             threadManager_.addThread(Component::Algorithm,
//                 std::thread(&SystemCaptureFactory::algorithmLoop, this)
//             );
//         } else {
//             spdlog::info("[SystemCaptureFactory] Algorithm not started (disabled by captureSelection).");
//         }

//         // Note: We do NOT spawn display thread because we want it on the main thread

//     } catch (const std::exception& e) {
//         reportError(fmt::format("Initialization failed: {}", e.what()));
//         stopCapture(); // Cleanup
//         throw;
//     }
// }




// inline void SystemCaptureFactory::stopCapture() {
//     running_ = false;

//     spdlog::info("[SystemCaptureFactory] Stopping camera threads...");
//     // If you used separate names like "Camera0", "Camera1", do:
//     // threadManager_.joinThreadsFor("Camera0");
//     // threadManager_.joinThreadsFor("Camera1");
//     // Or if using the enum:
//     // threadManager_.joinThreadsFor(Component::Camera);

//     spdlog::info("[SystemCaptureFactory] Stopping algorithm thread...");
//     // threadManager_.joinThreadsFor(Component::Algorithm);

//     spdlog::info("[SystemCaptureFactory] Stopping SoC monitoring thread...");
//     // threadManager_.joinThreadsFor("SoCMonitor");
//     // or .joinThreadsFor(Component::Custom);

//     // Finally, or optionally, do joinAll() if anything is left:
//     threadManager_.joinAll();

//     spdlog::info("[SystemCaptureFactory] Capture system stopped.");
// }

// inline void SystemCaptureFactory::resetDevices() {
//     for (auto& cam : cameras_) {
//         if (auto dataPtr = dynamic_cast<DataConcrete*>(cam.get())) {
//             dataPtr->resetDevice();
//         }
//     }
//     cameras_.clear();
//     algo_.reset();
//     soc_.reset();
//     display_.reset();
//     spdlog::info("[SystemCaptureFactory] All devices reset.");
// }

// inline void SystemCaptureFactory::pauseAll() {
//     paused_ = true;
//     spdlog::info("[SystemCaptureFactory] System paused.");
// }

// inline void SystemCaptureFactory::resumeAll() {
//     paused_ = false;
//     spdlog::info("[SystemCaptureFactory] System resumed.");
// }

// inline void SystemCaptureFactory::setDisplay(std::shared_ptr<IDisplay> display) {
//     display_ = std::move(display);
// }

// inline std::shared_ptr<IDisplay> SystemCaptureFactory::getDisplay() const {
//     return display_;
// }

// // --- Configuration Methods ---
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
//         reportError("Algorithm reconfiguration encountered issues.");
//     }
//     spdlog::info("[SystemCaptureFactory] Algorithm re-configured.");
// }

// inline void SystemCaptureFactory::configureSoC(const SoCConfig& cfg) {
//     if (soc_ && !soc_->configure(cfg)) {
//         reportError("SoC reconfiguration encountered issues.");
//     }
//     spdlog::info("[SystemCaptureFactory] SoC re-configured.");
// }

// inline void SystemCaptureFactory::configureDisplay(const DisplayConfig& cfg) {
//     if (display_) {
//         display_->configure(cfg);
//     }
//     spdlog::info("[SystemCaptureFactory] Display re-configured.");
// }

// // --- Error Handling ---
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

// // --- Accessor Methods ---
// inline std::shared_ptr<ISoC> SystemCaptureFactory::getSoC() const {
//     return soc_;
// }

// inline std::shared_ptr<IData> SystemCaptureFactory::getCamera(int index) const {
//     if (index >= 0 && static_cast<size_t>(index) < cameras_.size()) {
//         return cameras_[index];
//     }
//     return nullptr;
// }

// inline std::shared_ptr<IAlgorithm> SystemCaptureFactory::getAlgorithm() const {
//     return algo_;
// }

// // --- Thread Routines ---
// // Algorithm loop: simply sleeps while running.
// inline void SystemCaptureFactory::algorithmLoop() {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(50));
//         } else {
//             std::this_thread::sleep_for(std::chrono::milliseconds(10));
//         }
//     }
//     spdlog::info("[SystemCaptureFactory] Algorithm loop ended.");
// }

// // SoC monitoring loop.
// inline void SystemCaptureFactory::SoCLoop(ISoC* soc) {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(500));
//             continue;
//         }
//         auto perf = soc->getPerformance();
//         spdlog::info("[SoC] CPU1 {}%@{}MHz, CPU {}°C, GPU {}°C",
//                      perf.CPU1_Utilization_Percent,
//                      perf.CPU1_Frequency_MHz,
//                      perf.CPU_Temperature_C,
//                      perf.GPU_Temperature_C);
//         std::this_thread::sleep_for(std::chrono::seconds(1));
//     }
//     spdlog::info("[SystemCaptureFactory] SoC monitoring loop ended.");
// }

// // Capture loop: dequeues a frame, pushes it to the display queue, then re-queues the buffer.
// inline void SystemCaptureFactory::captureLoop(IData* camera) {
//     if (!camera) {
//         spdlog::error("[SystemCaptureFactory] Capture thread received null camera instance!");
//         return;
//     }
//     spdlog::info("[SystemCaptureFactory] Capture loop started for camera.");
//     while (running_) {
//         spdlog::debug("[SystemCaptureFactory] Capture loop iteration.");
//         void* dataPtr = nullptr;
//         size_t sizeBytes = 0;
//         size_t bufferIndex = 0;
//         if (!camera->dequeFrame(dataPtr, sizeBytes, bufferIndex)) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(5));
//             continue;
//         }
//         spdlog::debug("[SystemCaptureFactory] Frame captured ({} bytes)", sizeBytes);
        
//         // Create a FrameData object and copy the captured data.
//         FrameData frame;
//         frame.dataVec.assign(static_cast<uint8_t*>(dataPtr),
//                              static_cast<uint8_t*>(dataPtr) + sizeBytes);
//         frame.size = sizeBytes;
//         frame.width = cameraConfig_.width;
//         frame.height = cameraConfig_.height;
        
//         // Push the frame into the display original queue.
//         //displayOrigQueue_->push(frame);
//         //spdlog::debug("[SystemCaptureFactory] Frame pushed to display queue.");

//         spdlog::info("Queue addresses - Orig: {}, Proc: {}", 
//         (void*)displayOrigQueue_.get(), (void*)processedQueue_.get());
        
//         // Re-queue the buffer for continuous capture.
//         camera->queueBuffer(bufferIndex);
//     }
//     spdlog::info("[SystemCaptureFactory] Capture loop ended.");
// }

// inline bool SystemCaptureFactory::isComponentEnabled(const std::vector<Component>& components, Component c) {
//     return (std::find(components.begin(), components.end(), c) != components.end());
// }


// #endif // SYSTEMCAPTUREFACTORY_NEW_H


//===================================================================================================
// // systemcapturefactory_new.h

// #ifndef SYSTEMCAPTUREFACTORY_NEW_H
// #define SYSTEMCAPTUREFACTORY_NEW_H

// #include <atomic>
// //#include <thread>
// #include <vector>
// #include <memory>
// #include <unordered_set>
// #include <chrono>
// #include <tuple>
// #include <condition_variable>
// #include <spdlog/spdlog.h>
// #include <SDL2/SDL.h>
// #include <fmt/format.h>

// // Include interfaces
// #include "../Interfaces/IData.h"
// #include "../Interfaces/IAlgorithm.h"
// #include "../Interfaces/ISoC.h"
// #include "../Interfaces/IDisplay.h"
// #include "../Interfaces/ISystemProfiling.h"

// // Include shared structures and SharedQueue
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/AlgorithmConfig.h"
// #include "../SharedStructures/SoCConfig.h"
// #include "../SharedStructures/DisplayConfig.h"
// #include "../SharedStructures/SharedQueue.h"

// // include thread magager

// #include "../SharedStructures/ThreadManager.h"



// // Include concrete implementations (the _new versions)
// #include "../Concretes/DataConcrete_new.h"         // Updated: constructor now takes std::shared_ptr<SharedQueue<FrameData>>
// #include "../Concretes/AlgorithmConcrete_new.h"    // Updated: constructor now takes std::shared_ptr<SharedQueue<FrameData>>
// #include "../Concretes/SdlDisplayConcrete_new.h"     // Updated: constructor now takes std::shared_ptr<SharedQueue<FrameData>>
// #include "../Concretes/SoCConcrete.h"


// /**
//  * @class SystemCaptureFactory
//  * @brief Coordinates the main capture system in an asynchronous pipeline.
//  *
//  * In this design, DataConcrete pushes frames into two SharedQueue objects (one for the algorithm
//  * and one for displaying the original frame), AlgorithmConcrete reads from its input queue and
//  * pushes processed frames into a processed queue, and SdlDisplayConcrete reads from both queues.
//  */
// class SystemCaptureFactory : public ISystemProfiling {
// public:
//     // Constructors: one for a single camera, one for two cameras.
//     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                          std::shared_ptr<IAlgorithm> algo,
//                          std::shared_ptr<IData> camera0,
//                          std::shared_ptr<IData> camera1);
//     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                          std::shared_ptr<IAlgorithm> algo,
//                          std::shared_ptr<IData> camera);
//     ~SystemCaptureFactory() override;

//      // Store queues to close them later
//    // this->queues_ = {algoQueue, displayOrigQueue, processedQueue};

//     // ISystemProfiling interface methods
//     JetsonNanoInfo getSoCMetrics() const override;
//     std::tuple<double, int> getCameraMetrics(int cameraIndex) const override;
//     std::tuple<double, double> getAlgorithmMetrics() const override;

//     // System control methods
//     void initializeCapture(const CameraConfig& cameraConfig,
//                            const AlgorithmConfig& algoConfig);
//     void stopCapture();
//     void resetDevices();
//     void pauseAll();
//     void resumeAll();
//     void setDisplay(std::shared_ptr<IDisplay> display);

 

//     std::shared_ptr<IDisplay> getDisplay() const ;


//     // Configuration methods
//     void configureCamera(size_t index, const CameraConfig& cfg);
//     void configureAlgorithm(const AlgorithmConfig& cfg);
//     void configureSoC(const SoCConfig& cfg);
//     void configureDisplay(const DisplayConfig& cfg);

//     // Error handling
//     void setGlobalErrorCallback(std::function<void(const std::string&)>);

//     // Accessors for components
//     std::shared_ptr<ISoC> getSoC() const;
//     std::shared_ptr<IData> getCamera(int index = 0) const;
//     std::shared_ptr<IAlgorithm> getAlgorithm() const;
//     //std::shared_ptr<IDisplay> getDisplay() const;


// private:
//     // Thread routines
//     void captureLoop(IData* camera);
//     void algorithmLoop();
//     void displayLoop();
//     void SoCLoop(ISoC* soc);
//     void reportError(const std::string& msg);

// private:
//     std::atomic<bool> running_;
//     std::atomic<bool> paused_;
//     std::shared_ptr<ISoC>      soc_;
//     std::shared_ptr<IAlgorithm> algo_;
//     std::vector<std::shared_ptr<IData>> cameras_;  // One or more cameras
//     std::shared_ptr<IDisplay>  display_;           // Optional display component

//     std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue_; // Ensure this is initialized

//     // Threads

//     //Centralized Thread Management (ThreadManager)
//     ThreadManager threadManager_; // Replace vector<std::thread>

//     // Global error callback
//     std::function<void(const std::string&)> errorCallback_;

//     // Display helpers
//     int displayWidth_  = 640;
//     int displayHeight_ = 480;

//     // Configuration
//     CameraConfig cameraConfig_;

//     // Mutex for buffer tracking (if needed)
//     std::mutex bufferMutex_;
//     std::unordered_set<size_t> activeBuffers_;  // Tracks buffers in use


//     // Replace sleep_for() with Condition Variables
//     std::mutex mutex_;
//     std::condition_variable cv_;
//     bool processing_ = false;
// };

// // ===================== Implementation =====================

// inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                                                     std::shared_ptr<IAlgorithm> algo,
//                                                     std::shared_ptr<IData> camera0,
//                                                     std::shared_ptr<IData> camera1)
//     : running_(false), paused_(false), soc_(std::move(soc)), algo_(std::move(algo))
// {
//     if (!camera0 || !camera1) {
//         throw std::invalid_argument("[SystemCaptureFactory] Null camera pointer(s).");
//     }
//     cameras_.push_back(std::move(camera0));
//     cameras_.push_back(std::move(camera1));
// }

// inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                                                     std::shared_ptr<IAlgorithm> algo,
//                                                     std::shared_ptr<IData> camera)
//     : running_(false), paused_(false), soc_(std::move(soc)), algo_(std::move(algo))
// {
//     if (!camera) {
//         throw std::invalid_argument("[SystemCaptureFactory] Null camera pointer.");
//     }
//     cameras_.push_back(std::move(camera));
// }

// inline SystemCaptureFactory::~SystemCaptureFactory() {
//     stopCapture();
// }

// inline JetsonNanoInfo SystemCaptureFactory::getSoCMetrics() const {
//     return soc_ ? soc_->getPerformance() : JetsonNanoInfo{};
// }

// inline std::tuple<double, int> SystemCaptureFactory::getCameraMetrics(int cameraIndex) const {
//     if (cameraIndex < 0 || static_cast<size_t>(cameraIndex) >= cameras_.size()) {
//         spdlog::warn("[SystemCaptureFactory] Invalid camera index: {}", cameraIndex);
//         return {0.0, 0};
//     }
//     double fps = cameras_[cameraIndex]->getLastFPS();
//     int qSize  = cameras_[cameraIndex]->getQueueSize();
//     spdlog::debug("[SystemCaptureFactory] getCameraMetrics index={}, fps={}, queue={}",
//                   cameraIndex, fps, qSize);
//     return {fps, qSize};
// }

// inline std::tuple<double, double> SystemCaptureFactory::getAlgorithmMetrics() const {
//     if (!algo_) {
//         return {0.0, 0.0};
//     }
//     double fps = algo_->getFps();
//     double pt  = algo_->getAverageProcTime();
//     spdlog::debug("[SystemCaptureFactory] getAlgorithmMetrics fps={}, procTime={}", fps, pt);
//     return {fps, pt};
// }

// inline void SystemCaptureFactory::initializeCapture(const CameraConfig& cameraConfig,
//                                                       const AlgorithmConfig& algoConfig)
// {
//     try {
//         spdlog::info("[SystemCaptureFactory] Initializing capture system...");
//         cameraConfig_ = cameraConfig;

//         //Adjust these if you want to dynamically set from camera config
//         const int capWidth  = 640;
//         const int capHeight = 480;
//         std::vector<uint8_t> rgbOriginal(capWidth * capHeight * 3, 0);

    

//         // Open and configure each camera.
//         for (size_t i = 0; i < cameras_.size(); ++i) {
//         //for (auto& cam : cameras_) {
//             spdlog::info("[SystemCaptureFactory] Starting streaming for camera {}...",i);
//             //auto devicePath = fmt::format("/dev/video{}", i);
//             auto devicePath = fmt::format("/dev/video{}", i);
//             //spdlog::info("[SystemCaptureFactory] Opening camera at {}", devicePath);
//             if (!cameras_[i]->openDevice(devicePath)) {
//                 reportError(fmt::format("Failed to open {}", devicePath));
//                 continue;
//             }
//             if (!cameras_[i]->configure(cameraConfig_)) {
//                 reportError(fmt::format("Failed to configure {}", devicePath));
//                 continue;
//             }
//         }

//         // Start streaming on all cameras.
//         for (size_t i = 0; i < cameras_.size(); ++i) {
//          //for (auto& cam : cameras_) {
//             if (!cameras_[i]->startStreaming()) {
//                // reportError(fmt::format("Failed to start streaming for camera {}", i));
//                 continue;
//             }
//            // spdlog::info("[SystemCaptureFactory] Opening camera:  {}", i);
//         }

//         // Configure the algorithm.
//         if (algo_ && !algo_->configure(algoConfig)) {
//             reportError("Algorithm configuration failed.");
//         }

//         // Initialize SoC.
//         if (soc_) {
//             soc_->initializeSoC();
//         }

//         // If a display is provided, configure it.
//       // If a display is provided, configure it with valid settings.
//     if (display_) {
//         DisplayConfig dispCfg;
//         dispCfg.width = cameraConfig.width;  // Use camera's width/height
//         dispCfg.height = cameraConfig.height;
//         dispCfg.fullScreen = false;
//         dispCfg.windowTitle = "Framework Display";  // Set a valid title
//         display_->configure(dispCfg);  // Pass actual config, not empty
//         display_->initializeDisplay(dispCfg.width, dispCfg.height);
//         spdlog::info("[SystemCaptureFactory] Opening Framework Display{} {}", dispCfg.width, dispCfg.height);
//      } else {
//             spdlog::error("No display instance found!");
//         }


//         running_ = true;
//         paused_ = false;

//         // Centralized Thread Management (ThreadManager)
//         // Start SoC monitoring thread.
//         spdlog::info("[SystemCaptureFactory] Spawning SoC monitoring thread...");
//         //threadManager_.addThread(std::thread(&SystemCaptureFactory::SoCLoop, this, soc_.get()));
//         threadManager_.addThread(std::thread([this]() {
//         this->SoCLoop(soc_.get());  // Convert shared_ptr to raw pointer for direct access
//          }));

        
//         //std::vector<std::shared_ptr<IData>> cameras_;  // One or more cameras

//         // Start camera capture threads
//         // Now loop over the member variable:
//         // Start streaming on all cameras using ThreadManager
//         for (size_t i = 0; i < cameras_.size(); ++i) {
//             if (!cameras_[i]) {
//                 spdlog::error("[SystemCaptureFactory] Camera {} is null, skipping...", i);
//                 continue;
//             }

//             spdlog::info("[SystemCaptureFactory] Spawning capture thread for camera {}", i);

//             threadManager_.addThread(std::thread([this, i]() {
//                 this->captureLoop(cameras_[i].get());
//             }));
//         }



//         // // Fix incorrect shared_ptr capture
//         // std::thread socThread([this]() {
//         //     this->SoCLoop(soc_.get());  // Convert shared_ptr to raw pointer
//         // });
    

//         // // Add to ThreadManager to handle cleanup
//         // threadManager_.addThread(std::move(socThread));
    


//         // Start algorithm thread.
//         threadManager_.addThread(std::thread(&SystemCaptureFactory::algorithmLoop, this));      // Start algorithm thread.
//         // Start display thread.
//         threadManager_.addThread(std::thread(&SystemCaptureFactory::displayLoop, this));        // Start display thread.

//     }
//     catch (const std::exception& e) {
//         reportError(fmt::format("Initialization failed: {}", e.what()));
//         stopCapture();
//         throw;
//     }
// }

// inline void SystemCaptureFactory::stopCapture() {
//     running_ = false;
//     spdlog::info("[SystemCaptureFactory] Stopping all capture threads...");
//     threadManager_.joinAll(); // Join all threads USING Centralized Thread Management (ThreadManager)


//     // --- NOTE: The calls to stop() on the SharedQueues have been removed here
//     // because the component classes do not provide public getters for the queues.
//     // If you add public getters (e.g. getAlgoQueue(), getDisplayQueue(), getProcessedQueue()),
//     // then you can call stop() on them here.
//     //
//     // Example:
//     // if (auto dataPtr = dynamic_cast<DataConcrete*>(cameras_[0].get()))
//     //     dataPtr->getAlgoQueue()->stop();
//     // if (auto algPtr = dynamic_cast<AlgorithmConcrete*>(algo_.get()))
//     //     algPtr->getProcessedQueue()->stop();
//     //
//     // For now, these calls are commented out.
//     //
//     // End NOTE.
    
//     spdlog::info("[SystemCaptureFactory] Capture system stopped.");
// }

// inline void SystemCaptureFactory::resetDevices() {
//     for (auto& cam : cameras_) {
//         if (auto dataPtr = dynamic_cast<DataConcrete*>(cam.get())) {
//             dataPtr->resetDevice();
//         }
//     }
//     cameras_.clear();
//     algo_.reset();
//     soc_.reset();
//     display_.reset();
//     spdlog::info("[SystemCaptureFactory] All devices reset.");
// }

// inline void SystemCaptureFactory::pauseAll() {
//     paused_ = true;
//     spdlog::info("[SystemCaptureFactory] System paused.");
// }

// inline void SystemCaptureFactory::resumeAll() {
//     paused_ = false;
//     spdlog::info("[SystemCaptureFactory] System resumed.");
// }

// inline std::shared_ptr<IDisplay> SystemCaptureFactory::getDisplay() const {
//     return display_;
// }

// inline void SystemCaptureFactory::setDisplay(std::shared_ptr<IDisplay> display) {
//     display_ = std::move(display);
// }

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
//         reportError("Algorithm reconfiguration encountered issues.");
//     }
//     spdlog::info("[SystemCaptureFactory] Algorithm re-configured.");
// }

// inline void SystemCaptureFactory::configureSoC(const SoCConfig& cfg) {
//     if (soc_ && !soc_->configure(cfg)) {
//         reportError("SoC reconfiguration encountered issues.");
//     }
//     spdlog::info("[SystemCaptureFactory] SoC re-configured.");
// }

// inline void SystemCaptureFactory::configureDisplay(const DisplayConfig& cfg) {
//     if (display_) {
//         display_->configure(cfg);
//     }
//     spdlog::info("[SystemCaptureFactory] Display re-configured.");
// }

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

// inline std::shared_ptr<ISoC> SystemCaptureFactory::getSoC() const {
//     return soc_;
// }

// inline std::shared_ptr<IData> SystemCaptureFactory::getCamera(int index) const {
//     if (index >= 0 && static_cast<size_t>(index) < cameras_.size()) {
//         return cameras_[index];
//     }
//     return nullptr;
// }

// inline std::shared_ptr<IAlgorithm> SystemCaptureFactory::getAlgorithm() const {
//     return algo_;
// }

// // -------------------- Thread Routines --------------------

// // inline void SystemCaptureFactory::captureLoop(IData* camera) {
// //     std::unique_lock<std::mutex> lock(mutex_);
// //     while (running_) {
// //         spdlog::info("[SystemCaptureFactory] CaptureLoop Running.");
// //         if (paused_) {
// //             cv_.wait(lock, [this] { return processing_; });
// //             continue;
// //         }
// //         void* dataPtr = nullptr;
// //         size_t sizeBytes = 0;
// //         size_t bufferIndex = 0;
// //         if (!camera->dequeFrame(dataPtr, sizeBytes, bufferIndex)) {
// //             std::this_thread::sleep_for(std::chrono::milliseconds(5));
// //             continue;
// //         }
// //         {
// //             std::lock_guard<std::mutex> lock(bufferMutex_);
// //             activeBuffers_.insert(bufferIndex);
            
            
// //         }
// //         // DataConcrete is responsible for pushing the frame into its two SharedQueues.
// //         processing_ = false;
// //     }
// // }

// inline void SystemCaptureFactory::algorithmLoop() {
//     // In this design, AlgorithmConcrete spawns its own processing thread internally.
//     // This thread can be used for additional monitoring if desired.
//     while (running_) {
//         //spdlog::info("[SystemCaptureFactory] AlgorithmLoopRunning.");
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(50));
//         } else {
//             std::this_thread::sleep_for(std::chrono::milliseconds(10));
//         }
//     }
//     spdlog::info("[SystemCaptureFactory] Algorithm loop ended.");
// }

// inline void SystemCaptureFactory::SoCLoop(ISoC* soc) {
    
//     while (running_) {
//         //spdlog::info("[SystemCaptureFactory] SoC monitoring loop started.");
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(500));
//             continue;
//         }
//         auto perf = soc->getPerformance();
//         spdlog::info("[SoC] CPU1 {}%@{}MHz, CPU {}°C, GPU {}°C",
//                      perf.CPU1_Utilization_Percent,
//                      perf.CPU1_Frequency_MHz,
//                      perf.CPU_Temperature_C,
//                      perf.GPU_Temperature_C);
//         std::this_thread::sleep_for(std::chrono::seconds(1));
//     }
//     spdlog::info("[SystemCaptureFactory] SoC monitoring loop ended.");
// }

// // inline void SystemCaptureFactory::SoCLoop(ISoC* soc) {
// //     if (!soc) {
// //         spdlog::error("[SystemCaptureFactory] SoC monitoring received null pointer!");
// //         return;
// //     }

// //     spdlog::info("[SystemCaptureFactory] SoC monitoring loop started.");

// //     while (running_) {
// //         if (paused_) {
// //             std::this_thread::sleep_for(std::chrono::milliseconds(500));  // Reduce CPU load while paused
// //             continue;
// //         }

// //         // Retrieve SoC performance metrics
// //         auto perf = soc->getPerformance();
// //         spdlog::info("[SoC] CPU1 {}%@{}MHz, CPU {}°C, GPU {}°C",
// //                      perf.CPU1_Utilization_Percent,
// //                      perf.CPU1_Frequency_MHz,
// //                      perf.CPU_Temperature_C,
// //                      perf.GPU_Temperature_C);

// //         std::this_thread::sleep_for(std::chrono::seconds(1));  // Update every second
// //     }

// //     spdlog::info("[SystemCaptureFactory] SoC monitoring loop ended.");
// // }




// inline void SystemCaptureFactory::captureLoop(IData* camera) {
//     if (!camera) {
//         spdlog::error("[SystemCaptureFactory] Capture thread received null camera instance!");
//         return;
//     }
//     spdlog::info("[SystemCaptureFactory] Capture loop started for camera.");
//     while (running_) {
//         spdlog::debug("[SystemCaptureFactory] Capture loop started.");
//         void* dataPtr = nullptr;
//         size_t sizeBytes = 0;
//         size_t bufferIndex = 0;

//         if (!camera->dequeFrame(dataPtr, sizeBytes, bufferIndex)) {

//             std::this_thread::sleep_for(std::chrono::milliseconds(5));
//             continue;
//         }
//         // (Optionally) process the frame or simply requeue it:
//          spdlog::debug("[SystemCaptureFactory] Frame captured ({} bytes)", sizeBytes);
        
//         // Push frame to the display queue
//         FrameData frame;
//         frame.dataVec.assign(static_cast<uint8_t*>(dataPtr), static_cast<uint8_t*>(dataPtr) + sizeBytes);
//         frame.size = sizeBytes;
//         frame.width = cameraConfig_.width;
//         frame.height = cameraConfig_.height;

//         displayOrigQueue_->push(frame);
//         spdlog::debug("[SystemCaptureFactory] Frame pushed to display queue.");

//         // Re-queue the buffer after processing
//         camera->queueBuffer(bufferIndex);
        

//          }
//          spdlog::info("[SystemCaptureFactory] Capture loop ended.");
// }


// inline void SystemCaptureFactory::displayLoop() {
//     spdlog::debug("[SystemCaptureFactory] Display loop active.");
//     while (running_) {
//       //  spdlog::info("[SystemCaptureFactory] Display loop running.");
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(50));
//             continue;
//         }
//         if (display_) {
//             display_->renderAndPollEvents();
//         } else {
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//             spdlog::error("Display instance is null.");
//         }
//         std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS refresh
//     }
//     spdlog::info("[SystemCaptureFactory] Display loop ended.");
// }

// #endif // SYSTEMCAPTUREFACTORY_NEW_H


// #ifndef SYSTEMCAPTUREFACTORY_NEW_H
// #define SYSTEMCAPTUREFACTORY_NEW_H

// #include <atomic>
// #include <thread>
// #include <vector>
// #include <memory>
// #include <unordered_set>
// #include <chrono>
// #include <tuple>
// #include <condition_variable>
// #include <spdlog/spdlog.h>
// #include <SDL2/SDL.h>
// #include <fmt/format.h>

// // Include interfaces
// #include "../Interfaces/IData.h"
// #include "../Interfaces/IAlgorithm.h"
// #include "../Interfaces/ISoC.h"
// #include "../Interfaces/IDisplay.h"
// #include "../Interfaces/ISystemProfiling.h"

// // Include shared structures and SharedQueue
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/AlgorithmConfig.h"
// #include "../SharedStructures/SoCConfig.h"
// #include "../SharedStructures/DisplayConfig.h"
// #include "../SharedStructures/SharedQueue.h"

// // Include concrete implementations (the _new versions)
// #include "../Concretes/DataConcrete_new.h"         // Updated: now takes shared_ptr<SharedQueue<FrameData>> in constructor and provides public getters.
// #include "../Concretes/AlgorithmConcrete_new.h"    // Updated: now takes shared_ptr<SharedQueue<FrameData>> in constructor and provides getProcessedQueue()
// #include "../Concretes/SdlDisplayConcrete_new.h"     // Updated: now takes shared_ptr<SharedQueue<FrameData>> in constructor.
// #include "../Concretes/SoCConcrete.h"

// /**
//  * @class SystemCaptureFactory
//  * @brief Coordinates the main capture system in an asynchronous pipeline.
//  *
//  * In this design, DataConcrete pushes frames into two SharedQueue objects (one for the algorithm and one for displaying the original frame).  
//  * AlgorithmConcrete reads from its input queue and pushes processed frames into a third SharedQueue.  
//  * SdlDisplayConcrete continuously pops frames from both queues and renders them.
//  */
// class SystemCaptureFactory : public ISystemProfiling {
// public:
//     // Constructors: one for a single camera, one for two cameras.
//     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                          std::shared_ptr<IAlgorithm> algo,
//                          std::shared_ptr<IData> camera0,
//                          std::shared_ptr<IData> camera1);
//     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                          std::shared_ptr<IAlgorithm> algo,
//                          std::shared_ptr<IData> camera);
//     ~SystemCaptureFactory() override;

//     // ISystemProfiling interface methods
//     JetsonNanoInfo getSoCMetrics() const override;
//     std::tuple<double, int> getCameraMetrics(int cameraIndex) const override;
//     std::tuple<double, double> getAlgorithmMetrics() const override;

//     // System control methods
//     void initializeCapture(const CameraConfig& cameraConfig,
//                            const AlgorithmConfig& algoConfig);
//     void stopCapture();
//     void resetDevices();
//     void pauseAll();
//     void resumeAll();
//     void setDisplay(std::shared_ptr<IDisplay> display);

//     // Configuration methods
//     void configureCamera(size_t index, const CameraConfig& cfg);
//     void configureAlgorithm(const AlgorithmConfig& cfg);
//     void configureSoC(const SoCConfig& cfg);
//     void configureDisplay(const DisplayConfig& cfg);

//     // Error handling
//     void setGlobalErrorCallback(std::function<void(const std::string&)>);

//     // Accessors for components
//     std::shared_ptr<ISoC> getSoC() const;
//     std::shared_ptr<IData> getCamera(int index = 0) const;
//     std::shared_ptr<IAlgorithm> getAlgorithm() const;
//     std::shared_ptr<IDisplay> getDisplay() const;

// private:
//     // Thread routines
//     void captureLoop(IData* camera);
//     void algorithmLoop();
//     void displayLoop();
//     void SoCLoop(ISoC* soc);
//     void reportError(const std::string& msg);

// private:
//     std::atomic<bool> running_;
//     std::atomic<bool> paused_;
//     std::shared_ptr<ISoC>      soc_;
//     std::shared_ptr<IAlgorithm> algo_;
//     std::vector<std::shared_ptr<IData>> cameras_;  // One or more cameras
//     std::shared_ptr<IDisplay>  display_;           // Optional display component

//     // Threads
//     std::vector<std::thread> cameraThreads_;
//     std::thread algorithmThread_;
//     std::thread displayThread_;
//     std::thread socThread_;

//     // Global error callback
//     std::function<void(const std::string&)> errorCallback_;

//     // Display helpers
//     int displayWidth_  = 640;
//     int displayHeight_ = 480;

//     // Configuration
//     CameraConfig cameraConfig_;

//     // Mutex for buffer tracking (if needed)
//     std::mutex bufferMutex_;
//     std::unordered_set<size_t> activeBuffers_;  // Tracks buffers in use
// };

// // ===================== Implementation =====================

// inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                                                     std::shared_ptr<IAlgorithm> algo,
//                                                     std::shared_ptr<IData> camera0,
//                                                     std::shared_ptr<IData> camera1)
//     : running_(false), paused_(false), soc_(std::move(soc)), algo_(std::move(algo))
// {
//     if (!camera0 || !camera1) {
//         throw std::invalid_argument("[SystemCaptureFactory] Null camera pointer(s).");
//     }
//     cameras_.push_back(std::move(camera0));
//     cameras_.push_back(std::move(camera1));
// }

// inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                                                     std::shared_ptr<IAlgorithm> algo,
//                                                     std::shared_ptr<IData> camera)
//     : running_(false), paused_(false), soc_(std::move(soc)), algo_(std::move(algo))
// {
//     if (!camera) {
//         throw std::invalid_argument("[SystemCaptureFactory] Null camera pointer.");
//     }
//     cameras_.push_back(std::move(camera));
// }

// inline SystemCaptureFactory::~SystemCaptureFactory() {
//     stopCapture();
// }

// inline JetsonNanoInfo SystemCaptureFactory::getSoCMetrics() const {
//     return soc_ ? soc_->getPerformance() : JetsonNanoInfo{};
// }

// inline std::tuple<double, int> SystemCaptureFactory::getCameraMetrics(int cameraIndex) const {
//     if (cameraIndex < 0 || static_cast<size_t>(cameraIndex) >= cameras_.size()) {
//         spdlog::warn("[SystemCaptureFactory] Invalid camera index: {}", cameraIndex);
//         return {0.0, 0};
//     }
//     double fps = cameras_[cameraIndex]->getLastFPS();
//     int qSize  = cameras_[cameraIndex]->getQueueSize();
//     spdlog::debug("[SystemCaptureFactory] getCameraMetrics index={}, fps={}, queue={}",
//                   cameraIndex, fps, qSize);
//     return {fps, qSize};
// }

// inline std::tuple<double, double> SystemCaptureFactory::getAlgorithmMetrics() const {
//     if (!algo_) {
//         return {0.0, 0.0};
//     }
//     double fps = algo_->getFps();
//     double pt  = algo_->getAverageProcTime();
//     spdlog::debug("[SystemCaptureFactory] getAlgorithmMetrics fps={}, procTime={}", fps, pt);
//     return {fps, pt};
// }

// inline void SystemCaptureFactory::initializeCapture(const CameraConfig& cameraConfig,
//                                                       const AlgorithmConfig& algoConfig)
// {
//     try {
//         spdlog::info("[SystemCaptureFactory] Initializing capture system...");
//         cameraConfig_ = cameraConfig;

//         // Open and configure each camera.
//         for (size_t i = 0; i < cameras_.size(); ++i) {
//             auto devicePath = fmt::format("/dev/video{}", i);
//             spdlog::info("[SystemCaptureFactory] Opening camera at {}", devicePath);
//             if (!cameras_[i]->openDevice(devicePath)) {
//                 reportError(fmt::format("Failed to open {}", devicePath));
//                 continue;
//             }
//             if (!cameras_[i]->configure(cameraConfig_)) {
//                 reportError(fmt::format("Failed to configure {}", devicePath));
//                 continue;
//             }
//         }

//         // Start streaming on all cameras.
//         for (size_t i = 0; i < cameras_.size(); ++i) {
//             if (!cameras_[i]->startStreaming()) {
//                 reportError(fmt::format("Failed to start streaming for camera {}", i));
//                 continue;
//             }
//         }

//         // Configure the algorithm.
//         if (algo_ && !algo_->configure(algoConfig)) {
//             reportError("Algorithm configuration failed.");
//         }

//         // Initialize SoC.
//         if (soc_) {
//             soc_->initializeSoC();
//         }

//         // If a display is provided, configure it.
//         if (display_) {
//             configureDisplay(DisplayConfig{});
//             display_->initializeDisplay(displayWidth_, displayHeight_);
//         }

//         running_ = true;
//         paused_ = false;

//         // Start camera capture threads.
//         for (auto& cam : cameras_) {
//             cameraThreads_.emplace_back(&SystemCaptureFactory::captureLoop, this, cam.get());
//         }

//         // Start algorithm thread.
//         algorithmThread_ = std::thread(&SystemCaptureFactory::algorithmLoop, this);

//         // Start SoC monitoring thread.
//         socThread_ = std::thread(&SystemCaptureFactory::SoCLoop, this, soc_.get());

//         // Start display thread.
//         displayThread_ = std::thread(&SystemCaptureFactory::displayLoop, this);

//         spdlog::info("[SystemCaptureFactory] Capture system initialized with {} camera(s).", cameras_.size());
//     }
//     catch (const std::exception& e) {
//         reportError(fmt::format("Initialization failed: {}", e.what()));
//         stopCapture();
//         throw;
//     }
// }

// inline void SystemCaptureFactory::stopCapture() {
//     running_ = false;
//     // Join camera threads.
//     for (auto& t : cameraThreads_) {
//         if (t.joinable())
//             t.join();
//     }
//     cameraThreads_.clear();

//     if (algorithmThread_.joinable())
//         algorithmThread_.join();
//     if (socThread_.joinable())
//         socThread_.join();
//     if (displayThread_.joinable())
//         displayThread_.join();

//     // Stop each camera.
//     for (auto& cam : cameras_) {
//         cam->stopStreaming();
//     }

//     // --- Stop the SharedQueues to unblock any waiting pop() calls ---
//     if (!cameras_.empty()) {
//         // Assuming the first camera is our DataConcrete instance.
//         auto* dataPtr = dynamic_cast<DataConcrete*>(cameras_[0].get());
//         if (dataPtr) {
//             // Use the public getters (which now return shared_ptr) and call arrow operator.
//             dataPtr->getAlgoQueue()->stop();
//             dataPtr->getDisplayQueue()->stop();
//         }
//     }
//     if (algo_) {
//         auto* algPtr = dynamic_cast<AlgorithmConcrete*>(algo_.get());
//         if (algPtr) {
//             algPtr->getProcessedQueue()->stop();
//         }
//     }
//     spdlog::info("[SystemCaptureFactory] Capture system stopped.");
// }

// inline void SystemCaptureFactory::resetDevices() {
//     for (auto& cam : cameras_) {
//         auto dataPtr = dynamic_cast<DataConcrete*>(cam.get());
//         if (dataPtr) {
//             dataPtr->resetDevice();
//         }
//     }
//     cameras_.clear();
//     algo_.reset();
//     soc_.reset();
//     display_.reset();
//     spdlog::info("[SystemCaptureFactory] All devices reset.");
// }

// inline void SystemCaptureFactory::pauseAll() {
//     paused_ = true;
//     spdlog::info("[SystemCaptureFactory] System paused.");
// }

// inline void SystemCaptureFactory::resumeAll() {
//     paused_ = false;
//     spdlog::info("[SystemCaptureFactory] System resumed.");
// }

// inline void SystemCaptureFactory::setDisplay(std::shared_ptr<IDisplay> display) {
//     display_ = std::move(display);
// }

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
//         reportError("Algorithm reconfiguration encountered issues.");
//     }
//     spdlog::info("[SystemCaptureFactory] Algorithm re-configured.");
// }

// inline void SystemCaptureFactory::configureSoC(const SoCConfig& cfg) {
//     if (soc_ && !soc_->configure(cfg)) {
//         reportError("SoC reconfiguration encountered issues.");
//     }
//     spdlog::info("[SystemCaptureFactory] SoC re-configured.");
// }

// inline void SystemCaptureFactory::configureDisplay(const DisplayConfig& cfg) {
//     if (display_) {
//         display_->configure(cfg);
//     }
//     spdlog::info("[SystemCaptureFactory] Display re-configured.");
// }

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

// inline std::shared_ptr<ISoC> SystemCaptureFactory::getSoC() const {
//     return soc_;
// }

// inline std::shared_ptr<IData> SystemCaptureFactory::getCamera(int index) const {
//     if (index >= 0 && static_cast<size_t>(index) < cameras_.size()) {
//         return cameras_[index];
//     }
//     return nullptr;
// }

// inline std::shared_ptr<IAlgorithm> SystemCaptureFactory::getAlgorithm() const {
//     return algo_;
// }

// // -------------------- Thread Routines --------------------

// inline void SystemCaptureFactory::captureLoop(IData* camera) {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//             continue;
//         }
//         void* dataPtr = nullptr;
//         size_t sizeBytes = 0;
//         size_t bufferIndex = 0;
//         if (!camera->dequeFrame(dataPtr, sizeBytes, bufferIndex)) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(5));
//             continue;
//         }
//         {
//             std::lock_guard<std::mutex> lock(bufferMutex_);
//             activeBuffers_.insert(bufferIndex);
//         }
//         // DataConcrete is responsible for pushing frames into its two SharedQueues.
//     }
// }

// inline void SystemCaptureFactory::algorithmLoop() {
//     // In this design, AlgorithmConcrete spawns its own processing thread internally.
//     // This thread here can be used for additional monitoring if desired.
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(50));
//         } else {
//             std::this_thread::sleep_for(std::chrono::milliseconds(10));
//         }
//     }
//     spdlog::info("[SystemCaptureFactory] Algorithm loop ended.");
// }

// inline void SystemCaptureFactory::SoCLoop(ISoC* soc) {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(500));
//             continue;
//         }
//         auto perf = soc->getPerformance();
//         spdlog::info("[SoC] CPU1 {}%@{}MHz, CPU {}°C, GPU {}°C",
//                      perf.CPU1_Utilization_Percent,
//                      perf.CPU1_Frequency_MHz,
//                      perf.CPU_Temperature_C,
//                      perf.GPU_Temperature_C);
//         std::this_thread::sleep_for(std::chrono::seconds(1));
//     }
//     spdlog::info("[SystemCaptureFactory] SoC monitoring loop ended.");
// }

// inline void SystemCaptureFactory::displayLoop() {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(50));
//             continue;
//         }
//         if (display_) {
//             display_->renderAndPollEvents();
//         } else {
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//         }
//         std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS refresh
//     }
//     spdlog::info("[SystemCaptureFactory] Display loop ended.");
// }

// #endif // SYSTEMCAPTUREFACTORY_NEW_H



//===========================================
// #ifndef SYSTEMCAPTUREFACTORY_NEW_H
// #define SYSTEMCAPTUREFACTORY_NEW_H

// #include <atomic>
// #include <thread>
// #include <vector>
// #include <memory>
// #include <unordered_set>
// #include <chrono>
// #include <tuple>
// #include <condition_variable>
// #include <spdlog/spdlog.h>
// #include <SDL2/SDL.h>
// #include <fmt/format.h>

// // Include interfaces
// #include "../Interfaces/IData.h"
// #include "../Interfaces/IAlgorithm.h"
// #include "../Interfaces/ISoC.h"
// #include "../Interfaces/IDisplay.h"
// #include "../Interfaces/ISystemProfiling.h"

// // Include shared structures and SharedQueue
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/AlgorithmConfig.h"
// #include "../SharedStructures/SoCConfig.h"
// #include "../SharedStructures/DisplayConfig.h"
// #include "../SharedStructures/SharedQueue.h"

// // Include concrete implementations (the _new versions)
// #include "../Concretes/DataConcrete_new.h"
// #include "../Concretes/AlgorithmConcrete_new.h"
// #include "../Concretes/SdlDisplayConcrete_new.h"
// #include "../Concretes/SoCConcrete.h"

// /**
//  * @class SystemCaptureFactory
//  * @brief Coordinates the main capture system in an asynchronous pipeline.
//  *
//  * In this new design, we assume that the components have been constructed with
//  * the proper SharedQueue dependencies. DataConcrete pushes frames into two queues,
//  * AlgorithmConcrete processes frames and pushes to a processed queue, and SdlDisplayConcrete
//  * reads from both queues.
//  */
// class SystemCaptureFactory : public ISystemProfiling {
// public:
//     // Two constructors are provided: one accepting two cameras and one with a single camera.
//     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                          std::shared_ptr<IAlgorithm> algo,
//                          std::shared_ptr<IData> camera0,
//                          std::shared_ptr<IData> camera1);
//     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                          std::shared_ptr<IAlgorithm> algo,
//                          std::shared_ptr<IData> camera);
//     ~SystemCaptureFactory() override;

//     // ISystemProfiling interface methods
//     JetsonNanoInfo getSoCMetrics() const override;
//     std::tuple<double, int> getCameraMetrics(int cameraIndex) const override;
//     std::tuple<double, double> getAlgorithmMetrics() const override;

//     // System control methods
//     void initializeCapture(const CameraConfig& cameraConfig,
//                            const AlgorithmConfig& algoConfig);
//     void stopCapture();
//     void resetDevices();
//     void pauseAll();
//     void resumeAll();
//     void setDisplay(std::shared_ptr<IDisplay> display);

//     // Configuration methods
//     void configureCamera(size_t index, const CameraConfig& cfg);
//     void configureAlgorithm(const AlgorithmConfig& cfg);
//     void configureSoC(const SoCConfig& cfg);
//     void configureDisplay(const DisplayConfig& cfg);

//     // Error handling
//     void setGlobalErrorCallback(std::function<void(const std::string&)>);

//     // Accessors for components
//     std::shared_ptr<ISoC> getSoC() const;
//     std::shared_ptr<IData> getCamera(int index = 0) const;
//     std::shared_ptr<IAlgorithm> getAlgorithm() const;
//     std::shared_ptr<IDisplay> getDisplay() const;

// private:
//     // Thread routines
//     void captureLoop(IData* camera);
//     void algorithmLoop();
//     void displayLoop();
//     void SoCLoop(ISoC* soc);
//     void reportError(const std::string& msg);

// private:
//     std::atomic<bool> running_;
//     std::atomic<bool> paused_;
//     std::shared_ptr<ISoC>      soc_;
//     std::shared_ptr<IAlgorithm> algo_;
//     std::vector<std::shared_ptr<IData>> cameras_;  // Support for one or more cameras
//     std::shared_ptr<IDisplay>  display_;           // Optional display component

//     // Threads
//     std::vector<std::thread> cameraThreads_;
//     std::thread algorithmThread_;
//     std::thread displayThread_;
//     std::thread socThread_;

//     // In this new design, the shared queues are managed internally by the components.
//     // (For example, DataConcrete was constructed with its two queues and AlgorithmConcrete
//     // with its input and output queues.)

//     // Global error callback
//     std::function<void(const std::string&)> errorCallback_;

//     // Display helpers
//     int displayWidth_  = 640;
//     int displayHeight_ = 480;

//     // Configuration
//     CameraConfig cameraConfig_;

//     // Mutexes for buffer tracking (if needed)
//     std::mutex bufferMutex_;
//     std::unordered_set<size_t> activeBuffers_;  // Tracks active buffers
// };

// // ===================== Implementation =====================

// inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                                                     std::shared_ptr<IAlgorithm> algo,
//                                                     std::shared_ptr<IData> camera0,
//                                                     std::shared_ptr<IData> camera1)
//     : running_(false), paused_(false), soc_(std::move(soc)), algo_(std::move(algo))
// {
//     if (!camera0 || !camera1) {
//         throw std::invalid_argument("[SystemCaptureFactory] Null camera pointer(s).");
//     }
//     cameras_.push_back(std::move(camera0));
//     cameras_.push_back(std::move(camera1));
// }

// inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                                                     std::shared_ptr<IAlgorithm> algo,
//                                                     std::shared_ptr<IData> camera)
//     : running_(false), paused_(false), soc_(std::move(soc)), algo_(std::move(algo))
// {
//     if (!camera) {
//         throw std::invalid_argument("[SystemCaptureFactory] Null camera pointer.");
//     }
//     cameras_.push_back(std::move(camera));
// }

// inline SystemCaptureFactory::~SystemCaptureFactory() {
//     stopCapture();
// }

// inline JetsonNanoInfo SystemCaptureFactory::getSoCMetrics() const {
//     return soc_ ? soc_->getPerformance() : JetsonNanoInfo{};
// }

// inline std::tuple<double, int> SystemCaptureFactory::getCameraMetrics(int cameraIndex) const {
//     if (cameraIndex < 0 || static_cast<size_t>(cameraIndex) >= cameras_.size()) {
//         spdlog::warn("[SystemCaptureFactory] Invalid camera index: {}", cameraIndex);
//         return {0.0, 0};
//     }
//     double fps = cameras_[cameraIndex]->getLastFPS();
//     int qSize  = cameras_[cameraIndex]->getQueueSize();
//     spdlog::debug("[SystemCaptureFactory] getCameraMetrics index={}, fps={}, queue={}", cameraIndex, fps, qSize);
//     return {fps, qSize};
// }

// inline std::tuple<double, double> SystemCaptureFactory::getAlgorithmMetrics() const {
//     if (!algo_) {
//         return {0.0, 0.0};
//     }
//     double fps = algo_->getFps();
//     double pt  = algo_->getAverageProcTime();
//     spdlog::debug("[SystemCaptureFactory] getAlgorithmMetrics fps={}, procTime={}", fps, pt);
//     return {fps, pt};
// }

// inline void SystemCaptureFactory::initializeCapture(const CameraConfig& cameraConfig,
//                                                       const AlgorithmConfig& algoConfig)
// {
//     try {
//         spdlog::info("[SystemCaptureFactory] Initializing capture system...");
//         cameraConfig_ = cameraConfig;

//         // Open and configure each camera.
//         for (size_t i = 0; i < cameras_.size(); ++i) {
//             auto devicePath = fmt::format("/dev/video{}", i);
//             spdlog::info("[SystemCaptureFactory] Opening camera at {}", devicePath);
//             if (!cameras_[i]->openDevice(devicePath)) {
//                 reportError(fmt::format("Failed to open {}", devicePath));
//                 continue;
//             }
//             if (!cameras_[i]->configure(cameraConfig_)) {
//                 reportError(fmt::format("Failed to configure {}", devicePath));
//                 continue;
//             }
//         }

//         // Start streaming on all cameras.
//         for (size_t i = 0; i < cameras_.size(); ++i) {
//             if (!cameras_[i]->startStreaming()) {
//                 reportError(fmt::format("Failed to start streaming for camera {}", i));
//                 continue;
//             }
//         }

//         // Configure algorithm.
//         if (algo_ && !algo_->configure(algoConfig)) {
//             reportError("Algorithm configuration failed.");
//         }

//         // Initialize SoC.
//         if (soc_) {
//             soc_->initializeSoC();
//         }

//         // If a display is provided, configure it.
//         if (display_) {
//             configureDisplay(DisplayConfig{});
//             display_->initializeDisplay(displayWidth_, displayHeight_);
//         }

//         running_ = true;
//         paused_ = false;

//         // Start camera capture threads.
//         for (auto& cam : cameras_) {
//             cameraThreads_.emplace_back(&SystemCaptureFactory::captureLoop, this, cam.get());
//         }

//         // Start algorithm thread.
//         algorithmThread_ = std::thread(&SystemCaptureFactory::algorithmLoop, this);

//         // Start SoC monitoring thread.
//         socThread_ = std::thread(&SystemCaptureFactory::SoCLoop, this, soc_.get());

//         // Start display thread.
//         displayThread_ = std::thread(&SystemCaptureFactory::displayLoop, this);

//         spdlog::info("[SystemCaptureFactory] Capture system initialized with {} camera(s).", cameras_.size());
//     }
//     catch (const std::exception& e) {
//         reportError(fmt::format("Initialization failed: {}", e.what()));
//         stopCapture();
//         throw;
//     }
// }

// inline void SystemCaptureFactory::stopCapture() {
//     running_ = false;
//     // Join camera threads.
//     for (auto& t : cameraThreads_) {
//         if (t.joinable())
//             t.join();
//     }
//     cameraThreads_.clear();

//     if (algorithmThread_.joinable())
//         algorithmThread_.join();
//     if (socThread_.joinable())
//         socThread_.join();
//     if (displayThread_.joinable())
//         displayThread_.join();

//     // Stop each camera.
//     for (auto& cam : cameras_) {
//         cam->stopStreaming();
//     }

//     // --- MODIFIED: Stop SharedQueues to unblock any waiting pop() calls ---
//     if (!cameras_.empty()) {
//         // Assuming the first camera is DataConcrete.
//         auto* dataPtr = dynamic_cast<DataConcrete*>(cameras_[0].get());
//         if (dataPtr) {
//             dataPtr->getAlgoQueue().stop();
//             dataPtr->getDisplayQueue().stop();
//         }
//     }
//     if (algo_) {
//         auto* algPtr = dynamic_cast<AlgorithmConcrete*>(algo_.get());
//         if (algPtr) {
//             algPtr->getProcessedQueue().stop();
//         }
//     }
//     spdlog::info("[SystemCaptureFactory] Capture system stopped.");
// }

// inline void SystemCaptureFactory::resetDevices() {
//     for (auto& cam : cameras_) {
//         auto dataPtr = dynamic_cast<DataConcrete*>(cam.get());
//         if (dataPtr) {
//             dataPtr->resetDevice();
//         }
//     }
//     cameras_.clear();
//     algo_.reset();
//     soc_.reset();
//     display_.reset();
//     spdlog::info("[SystemCaptureFactory] All devices reset.");
// }

// inline void SystemCaptureFactory::pauseAll() {
//     paused_ = true;
//     spdlog::info("[SystemCaptureFactory] System paused.");
// }

// inline void SystemCaptureFactory::resumeAll() {
//     paused_ = false;
//     spdlog::info("[SystemCaptureFactory] System resumed.");
// }

// inline void SystemCaptureFactory::setDisplay(std::shared_ptr<IDisplay> display) {
//     display_ = std::move(display);
// }

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
//         reportError("Algorithm reconfiguration encountered issues.");
//     }
//     spdlog::info("[SystemCaptureFactory] Algorithm re-configured.");
// }

// inline void SystemCaptureFactory::configureSoC(const SoCConfig& cfg) {
//     if (soc_ && !soc_->configure(cfg)) {
//         reportError("SoC reconfiguration encountered issues.");
//     }
//     spdlog::info("[SystemCaptureFactory] SoC re-configured.");
// }

// inline void SystemCaptureFactory::configureDisplay(const DisplayConfig& cfg) {
//     if (display_) {
//         display_->configure(cfg);
//     }
//     spdlog::info("[SystemCaptureFactory] Display re-configured.");
// }

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

// inline std::shared_ptr<ISoC> SystemCaptureFactory::getSoC() const {
//     return soc_;
// }

// inline std::shared_ptr<IData> SystemCaptureFactory::getCamera(int index) const {
//     if (index >= 0 && static_cast<size_t>(index) < cameras_.size()) {
//         return cameras_[index];
//     }
//     return nullptr;
// }

// inline std::shared_ptr<IAlgorithm> SystemCaptureFactory::getAlgorithm() const {
//     return algo_;
// }

// // -------------------- Thread Routines --------------------

// inline void SystemCaptureFactory::captureLoop(IData* camera) {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//             continue;
//         }
//         void* dataPtr = nullptr;
//         size_t sizeBytes = 0;
//         size_t bufferIndex = 0;
//         if (!camera->dequeFrame(dataPtr, sizeBytes, bufferIndex)) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(5));
//             continue;
//         }
//         {
//             std::lock_guard<std::mutex> lock(bufferMutex_);
//             activeBuffers_.insert(bufferIndex);
//         }
//         // DataConcrete already pushes the frame into its two queues.
//     }
// }

// inline void SystemCaptureFactory::algorithmLoop() {
//     // In this design, AlgorithmConcrete spawns its own processing thread internally.
//     // We can simply sleep here or use this thread for additional monitoring.
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(50));
//         } else {
//             std::this_thread::sleep_for(std::chrono::milliseconds(10));
//         }
//     }
//     spdlog::info("[SystemCaptureFactory] Algorithm loop ended.");
// }

// inline void SystemCaptureFactory::SoCLoop(ISoC* soc) {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(500));
//             continue;
//         }
//         auto perf = soc->getPerformance();
//         spdlog::info("[SoC] CPU1 {}%@{}MHz, CPU {}°C, GPU {}°C",
//                      perf.CPU1_Utilization_Percent,
//                      perf.CPU1_Frequency_MHz,
//                      perf.CPU_Temperature_C,
//                      perf.GPU_Temperature_C);
//         std::this_thread::sleep_for(std::chrono::seconds(1));
//     }
//     spdlog::info("[SystemCaptureFactory] SoC monitoring loop ended.");
// }

// inline void SystemCaptureFactory::displayLoop() {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(50));
//             continue;
//         }
//         if (display_) {
//             display_->renderAndPollEvents();
//         } else {
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//         }
//         std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS refresh
//     }
//     spdlog::info("[SystemCaptureFactory] Display loop ended.");
// }

// #endif // SYSTEMCAPTUREFACTORY_NEW_H



// // // SystemCaptureFactory.h
// // #pragma once

// // #include <atomic>
// // #include <queue>
// // #include <thread>
// // #include <vector>
// // #include <functional>
// // #include <mutex>
// // #include <unordered_set>
// // #include <chrono>
// // #include <tuple>
// // #include <condition_variable>
// // #include <spdlog/spdlog.h>
// // #include <SDL2/SDL.h>

// // // Interfaces
// // #include "../Interfaces/IData.h"
// // #include "../Interfaces/IAlgorithm.h"
// // #include "../Interfaces/ISoC.h"
// // #include "../Interfaces/ISystemProfiling.h"
// // #include "../Interfaces/IDisplay.h"

// // // Shared Structures
// // #include "../SharedStructures/FrameData.h"
// // #include "../SharedStructures/CameraConfig.h"
// // #include "../SharedStructures/AlgorithmConfig.h"
// // #include "../SharedStructures/SoCConfig.h"
// // #include "../SharedStructures/DisplayConfig.h"
// // #include "../SharedStructures/SharedQueue.h"
// // #include "../SharedStructures/jetsonNanoInfo.h"

// // // Concretes
// // #include "../Concretes/SdlDisplayConcrete_new.h"
// // #include "../Concretes/DataConcrete_new.h"
// // #include "../Concretes/AlgorithmConcrete_new.h"


// // /**
// //  * @class SystemCaptureFactory
// //  * @brief Coordinates the main “capture system” in an asynchronous pipeline:
// //  *   - Multiple cameras (async streaming)
// //  *   - A shared frame queue (producer-consumer)
// //  *   - A separate algorithm thread pulling from the queue
// //  *   - A display thread (optional)
// //  *   - A SoC monitoring thread
// //  *
// //  * Implements ISystemProfiling to provide SoC, camera, and algorithm metrics.
// //  */
// // class SystemCaptureFactory : public ISystemProfiling
// // {
// // public:
// //     // Constructors
// //     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
// //                          std::shared_ptr<IAlgorithm> algo,
// //                          std::shared_ptr<IData> camera0,
// //                          std::shared_ptr<IData> camera1);
// //     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
// //                          std::shared_ptr<IAlgorithm> algo,
// //                          std::shared_ptr<IData> camera);
// //     ~SystemCaptureFactory() override;

// //     // ISystemProfiling interface
// //     JetsonNanoInfo getSoCMetrics() const override;
// //     std::tuple<double, int> getCameraMetrics(int cameraIndex) const override;
// //     std::tuple<double, double> getAlgorithmMetrics() const override;

// //     // System control methods
// //     void initializeCapture(const CameraConfig& cameraConfig,
// //                            const AlgorithmConfig& algoConfig);
// //     void stopCapture();
// //     void resetDevices();
// //     void pauseAll();
// //     void resumeAll();
// //     void setDisplay(std::shared_ptr<IDisplay> display);

// //     // Configuration methods
// //     void configureCamera(size_t index, const CameraConfig& cfg);
// //     void configureAlgorithm(const AlgorithmConfig& cfg);
// //     void configureSoC(const SoCConfig& cfg);
// //     void configureDisplay(const DisplayConfig& cfg);

// //     // Error handling
// //     void setGlobalErrorCallback(std::function<void(const std::string&)>);

// //     // Accessors for components
// //     std::shared_ptr<ISoC> getSoC() const;
// //     std::shared_ptr<IData> getCamera(int index = 0) const;
// //     std::shared_ptr<IAlgorithm> getAlgorithm() const;

// // private:
// //     // Thread routines
// //     void captureLoop(IData* camera);
// //     void SoCLoop(ISoC* soc);
// //     void algorithmLoop(IAlgorithm* alg);
// //     void displayLoop();
// //     void reportError(const std::string& msg);

// // private:
// //     // Control flags
// //     std::atomic<bool> running_;
// //     std::atomic<bool> paused_;

// //     // Subsystems (injected)
// //     std::shared_ptr<ISoC>      soc_;
// //     std::shared_ptr<IAlgorithm> algo_;
// //     std::vector<std::shared_ptr<IData>> cameras_;  ///< Support for multiple cameras
// //     std::shared_ptr<IDisplay>  display_;           ///< Optional display component

// //     // Threads
// //     std::vector<std::thread> cameraThreads_;  ///< One thread per camera
// //     std::thread socThread_;                   ///< SoC monitoring thread
// //     std::thread displayThread_;               ///< Display thread
// //     std::thread algorithmThread_;             ///< Algorithm consumer thread

// //     // Producer-Consumer queue (Camera → Algorithm)
// //     SharedQueue<FrameData> frameQueue_;

// //     // Global error callback
// //     std::function<void(const std::string&)> errorCallback_;

// //     // Display helpers
// //     int displayWidth_  = 640;
// //     int displayHeight_ = 480;
// //     std::mutex processedMutex_;
// //     std::vector<uint8_t> processedRGB_;  ///< Optional storage for processed frames
// //     std::vector<uint8_t> rgbBuffer_;       ///< For local YUYV→RGB conversion

// //     // Configuration
// //     CameraConfig cameraConfig_;

// //     // Mutexes
// //     std::mutex bufferMutex_;
// //     std::unordered_set<size_t> activeBuffers_;  // Tracks buffers in use

// //     // Maximum queue size to avoid unbounded memory usage
// //     static constexpr size_t MAX_QUEUE_SIZE = 10;
// // };

// // // =====================================================================
// // // Inline Implementations (could be separated into a .cpp file)
// // // =====================================================================

// // // ---------------------------------------------------------------------
// // // Constructors and Destructor
// // // ---------------------------------------------------------------------
// // inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
// //                                                     std::shared_ptr<IAlgorithm> algo,
// //                                                     std::shared_ptr<IData> camera0,
// //                                                     std::shared_ptr<IData> camera1)
// //     : soc_(std::move(soc))
// //     , algo_(std::move(algo))
// //     , running_(false)
// //     , paused_(false)
// //     , errorCallback_(nullptr)
// // {
// //     if (!camera0 || !camera1) {
// //         throw std::invalid_argument("[SystemCaptureFactory] Null camera pointer(s).");
// //     }
// //     cameras_.push_back(std::move(camera0));
// //     cameras_.push_back(std::move(camera1));

// //     if (cameras_.empty()) {
// //         throw std::invalid_argument("[SystemCaptureFactory] At least one camera is required.");
// //     }
// // }

// // inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
// //                                                     std::shared_ptr<IAlgorithm> algo,
// //                                                     std::shared_ptr<IData> camera)
// //     : soc_(std::move(soc))
// //     , algo_(std::move(algo))
// //     , running_(false)
// //     , paused_(false)
// //     , errorCallback_(nullptr)
// // {
// //     if (!camera) {
// //         throw std::invalid_argument("[SystemCaptureFactory] Null camera pointer.");
// //     }
// //     cameras_.push_back(std::move(camera));
// //     if (cameras_.empty()) {
// //         throw std::invalid_argument("[SystemCaptureFactory] At least one camera is required.");
// //     }
// // }

// // inline SystemCaptureFactory::~SystemCaptureFactory()
// // {
// //     stopCapture();
// // }

// // // ---------------------------------------------------------------------
// // // ISystemProfiling Methods
// // // ---------------------------------------------------------------------
// // inline JetsonNanoInfo SystemCaptureFactory::getSoCMetrics() const {
// //     return soc_ ? soc_->getPerformance() : JetsonNanoInfo{};
// // }

// // inline std::tuple<double, int> SystemCaptureFactory::getCameraMetrics(int cameraIndex) const {
// //     if (cameraIndex < 0 || static_cast<size_t>(cameraIndex) >= cameras_.size()) {
// //         spdlog::warn("[SystemCaptureFactory] Invalid camera index: {}", cameraIndex);
// //         return {0.0, 0};
// //     }
// //     double fps   = cameras_[cameraIndex]->getLastFPS();
// //     int qSize    = cameras_[cameraIndex]->getQueueSize();
// //     spdlog::debug("[SystemCaptureFactory] getCameraMetrics index={}, fps={}, queue={}", 
// //                   cameraIndex, fps, qSize);
// //     return {fps, qSize};
// // }

// // inline std::tuple<double, double> SystemCaptureFactory::getAlgorithmMetrics() const {
// //     if (!algo_) {
// //         return {0.0, 0.0};
// //     }
// //     double fps = algo_->getFps();
// //     double pt  = algo_->getAverageProcTime();
// //     spdlog::debug("[SystemCaptureFactory] getAlgorithmMetrics fps={}, procTime={}", fps, pt);
// //     return {fps, pt};
// // }

// // // ---------------------------------------------------------------------
// // // System Control Methods
// // // ---------------------------------------------------------------------
// // // SystemCaptureFactory.h
// // inline void SystemCaptureFactory::initializeCapture(
// //     const CameraConfig& cameraConfig, 
// //     const AlgorithmConfig& algoConfig)
// // {
// //     try {
// //         spdlog::info("Initializing capture system...");
// //         cameraConfig_ = cameraConfig;

// //         // Initialize all cameras first
// //         for (size_t i = 0; i < cameras_.size(); ++i) {
// //             auto& camera = cameras_[i];
// //             const auto device_path = fmt::format("/dev/video{}", i);
            
// //             if (!camera->openDevice(device_path)) {
// //                 reportError(fmt::format("Failed to open {}", device_path));
// //                 continue;
// //             }

// //             if (!camera->configure(cameraConfig_)) {
// //                 reportError(fmt::format("Configuration failed for {}", device_path));
// //                 continue;
// //             }
// //         }

// //         // Start streaming after all cameras are configured
// //         for (size_t i = 0; i < cameras_.size(); ++i) {
// //             if (!cameras_[i]->startStreaming()) {
// //                 reportError(fmt::format("Failed to start streaming for camera {}", i));
// //                 continue;
// //             }
// //         }

// //         // Start processing threads
// //         running_ = true;
        
// //         // Start camera capture threads
// //         for (auto& camera : cameras_) {
// //             cameraThreads_.emplace_back(&SystemCaptureFactory::captureLoop, this, camera.get());
// //         }

// //         // Rest of initialization...
// //         // 2) Configure the algorithm
// //         if (algo_ && !algo_->configure(algoConfig)) {
// //             spdlog::warn("[SystemCaptureFactory] Algorithm config returned issues.");
// //         }

// //         // 3) Initialize SoC
// //         if (soc_) {
// //             soc_->initializeSoC();
// //         }

// //         // 4) If we have a display, configure it
// //         if (display_) {
// //             configureDisplay(DisplayConfig{});
// //             display_->initializeDisplay(displayWidth_, displayHeight_);
// //         }

// //         //  // Start algorithm thread
// //         // if (algo_) {
// //         //     algorithmThread_ = std::thread(&SystemCaptureFactory::algorithmLoop, this, algo_.get());
// //         // }

// //         // // Start SoC monitoring thread
// //         // if (soc_) {
// //         //     socThread_ = std::thread(&SystemCaptureFactory::SoCLoop, this, soc_.get());
// //         // }

// //         // Start display thread
// //         displayThread_ = std::thread(&SystemCaptureFactory::displayLoop, this);

// //         spdlog::info("[SystemCaptureFactory] Capture system initialized with {} camera(s).", cameras_.size());

// //     }
// //     catch (const std::exception& e) {
// //         spdlog::critical("Initialization failed: {}", e.what());
// //         stopCapture();
// //         throw;
// //     }
// // }

// // ///=============================0
// // // inline void SystemCaptureFactory::initializeCapture(
// // //     const CameraConfig& cameraConfig, 
// // //     const AlgorithmConfig& algoConfig)
// // // {
// // //     try {
// // //         spdlog::info("[SystemCaptureFactory] Initializing capture system...");
// // //         cameraConfig_ = cameraConfig;

// // //         // 1) Open & configure each camera
// // //         for (size_t i = 0; i < cameras_.size(); ++i) {
// // //             auto devicePath = "/dev/video" + std::to_string(i);
// // //             spdlog::info("[SystemCaptureFactory] Opening camera at {}", devicePath);

// // //             if (!cameras_[i]->openDevice(devicePath)) {
// // //                 reportError("Failed to open " + devicePath);
// // //                 continue;
// // //             }
// // //             if (!cameras_[i]->configure(cameraConfig_)) {
// // //                 reportError("Failed to configure camera " + devicePath);
// // //             }
// // //         }

// // //         // 2) Configure the algorithm
// // //         if (algo_ && !algo_->configure(algoConfig)) {
// // //             spdlog::warn("[SystemCaptureFactory] Algorithm config returned issues.");
// // //         }

// // //         // 3) Initialize SoC
// // //         if (soc_) {
// // //             soc_->initializeSoC();
// // //         }

// // //         // 4) If we have a display, configure it
// // //         if (display_) {
// // //             configureDisplay(DisplayConfig{});
// // //             display_->initializeDisplay(displayWidth_, displayHeight_);
// // //         }

// // //         // 5) Start threads
// // //         running_ = true;

// // //         // Start camera threads
// // //         for (size_t i = 0; i < cameras_.size(); ++i) {
// // //             auto devicePath = "/dev/video" + std::to_string(i);
// // //             if (!cameras_[i]->startStreaming()) {
// // //                 reportError("Failed to start streaming camera " + devicePath);
// // //                 continue;
// // //             }
// // //             cameraThreads_.emplace_back(&SystemCaptureFactory::captureLoop, this, cameras_[i].get());
// // //         }

// // //         // Start algorithm thread
// // //         if (algo_) {
// // //             algorithmThread_ = std::thread(&SystemCaptureFactory::algorithmLoop, this, algo_.get());
// // //         }

// // //         // Start SoC monitoring thread
// // //         if (soc_) {
// // //             socThread_ = std::thread(&SystemCaptureFactory::SoCLoop, this, soc_.get());
// // //         }

// // //         // Start display thread
// // //         displayThread_ = std::thread(&SystemCaptureFactory::displayLoop, this);

// // //         spdlog::info("[SystemCaptureFactory] Capture system initialized with {} camera(s).", cameras_.size());
// // //     }
// // //     catch (const std::exception& e) {
// // //         spdlog::error("[SystemCaptureFactory] Initialization error: {}", e.what());
// // //         stopCapture();
// // //         throw;
// // //     }
// // // }



// // inline void SystemCaptureFactory::stopCapture()
// // {
// //     if (!running_) {
// //         return;
// //     }
// //     spdlog::info("[SystemCaptureFactory] Stopping capture system...");
// //     running_ = false;

// //     for (auto& t : cameraThreads_) {
// //         if (t.joinable()) {
// //             t.join();
// //         }
// //     }
// //     cameraThreads_.clear();

// //     if (algorithmThread_.joinable()) {
// //         algorithmThread_.join();
// //     }

// //     if (socThread_.joinable()) {
// //         socThread_.join();
// //     }

// //     if (displayThread_.joinable()) {
// //         displayThread_.join();
// //     }

// //     resetDevices();
// //     spdlog::info("[SystemCaptureFactory] Capture system stopped.");
// // }

// // inline void SystemCaptureFactory::resetDevices()
// // {
// //     for (auto& camera : cameras_) {
// //         auto dataPtr = dynamic_cast<DataConcrete*>(camera.get());
// //         if (dataPtr) {
// //             dataPtr->resetDevice();
// //         }
// //     }
// //     cameras_.clear();

// //     algo_.reset();
// //     soc_.reset();
// //     display_.reset();

// //     spdlog::info("[SystemCaptureFactory] All devices reset.");
// // }

// // inline void SystemCaptureFactory::pauseAll()
// // {
// //     paused_ = true;
// //     spdlog::info("[SystemCaptureFactory] System paused.");
// // }

// // inline void SystemCaptureFactory::resumeAll()
// // {
// //     paused_ = false;
// //     spdlog::info("[SystemCaptureFactory] System resumed.");
// // }

// // inline void SystemCaptureFactory::setDisplay(std::shared_ptr<IDisplay> display)
// // {
// //     display_ = std::move(display);
// // }

// // // ---------------------------------------------------------------------
// // // Configuration Methods
// // // ---------------------------------------------------------------------
// // inline void SystemCaptureFactory::configureCamera(size_t index, const CameraConfig& cfg)
// // {
// //     if (index >= cameras_.size()) {
// //         reportError("Invalid camera index: " + std::to_string(index));
// //         return;
// //     }
// //     if (!cameras_[index]->configure(cfg)) {
// //         reportError("Failed to configure camera index " + std::to_string(index));
// //     }
// //     spdlog::info("[SystemCaptureFactory] Camera {} re-configured.", index);
// // }

// // inline void SystemCaptureFactory::configureAlgorithm(const AlgorithmConfig& cfg)
// // {
// //     if (algo_ && !algo_->configure(cfg)) {
// //         spdlog::warn("[SystemCaptureFactory] Algorithm reconfiguration encountered issues.");
// //     }
// //     spdlog::info("[SystemCaptureFactory] Algorithm re-configured.");
// // }

// // inline void SystemCaptureFactory::configureSoC(const SoCConfig& cfg)
// // {
// //     if (soc_ && !soc_->configure(cfg)) {
// //         spdlog::warn("[SystemCaptureFactory] SoC reconfiguration encountered issues.");
// //     }
// //     spdlog::info("[SystemCaptureFactory] SoC re-configured.");
// // }

// // inline void SystemCaptureFactory::configureDisplay(const DisplayConfig& cfg)
// // {
// //     if (display_) {
// //         display_->configure(cfg);
// //     }
// //     spdlog::info("[SystemCaptureFactory] Display re-configured.");
// // }

// // // ---------------------------------------------------------------------
// // // Error Handling and Accessors
// // // ---------------------------------------------------------------------
// // inline void SystemCaptureFactory::setGlobalErrorCallback(std::function<void(const std::string&)> callback)
// // {
// //     errorCallback_ = std::move(callback);
// // }

// // inline void SystemCaptureFactory::reportError(const std::string& msg)
// // {
// //     if (errorCallback_) {
// //         errorCallback_(msg);
// //     } else {
// //         spdlog::error("[SystemCaptureFactory] {}", msg);
// //     }
// // }

// // inline std::shared_ptr<ISoC> SystemCaptureFactory::getSoC() const 
// // {
// //     return soc_;
// // }

// // inline std::shared_ptr<IData> SystemCaptureFactory::getCamera(int index) const 
// // {
// //     if (index >= 0 && static_cast<size_t>(index) < cameras_.size()) {
// //         return cameras_[index];
// //     }
// //     return nullptr;
// // }

// // inline std::shared_ptr<IAlgorithm> SystemCaptureFactory::getAlgorithm() const 
// // {
// //     return algo_;
// // }

// // // ---------------------------------------------------------------------
// // // Thread Routine: Capture Loop (using FrameData with std::vector<uint8_t>)
// // // No pixel conversion is performed here.
// // // ---------------------------------------------------------------------
// // // ---------------------------------------------------------------------
// // // Modified captureLoop with buffer tracking
// // // ---------------------------------------------------------------------
// // inline void SystemCaptureFactory::captureLoop(IData* camera)
// // {
// //     int64_t localFrameCounter = 0;

// //     while (running_) {
// //         try {
            
// //         if (paused_) {
// //             std::this_thread::sleep_for(std::chrono::milliseconds(100));
// //             continue;
// //         }

// //         // Dequeued pointer from the driver
// //         void* rawPtr = nullptr;
// //         size_t rawSize = 0;
// //         size_t bufferIndex = 0;

// //         // Attempt to dequeue with buffer index
// //         if (!camera->dequeFrame(rawPtr, rawSize, bufferIndex)) {
// //             spdlog::warn("[SystemCaptureFactory] Failed to dequeue frame");
// //             std::this_thread::sleep_for(std::chrono::milliseconds(5));
// //             continue;
// //         }

// //         // Track active buffer
// //         {
// //             std::lock_guard<std::mutex> lock(bufferMutex_);
// //             activeBuffers_.insert(bufferIndex);
// //         }

// //         // Create FrameData with buffer index
// //         FrameData frame;
// //         frame.bufferIndex = bufferIndex;
// //         if (rawPtr && rawSize > 0) {
// //             frame.dataVec.resize(rawSize);
// //             std::memcpy(frame.dataVec.data(), rawPtr, rawSize);
// //             frame.size = rawSize;
// //             frame.width = cameraConfig_.width;
// //             frame.height = cameraConfig_.height;
// //             //frame.timestamp = /* ... */;

// //             // e.g., store a timestamp
// //             frame.timestamp = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
// //                                 std::chrono::steady_clock::now().time_since_epoch()).count();
// //            // frameQueue_.push(frame);
// //         }

// //         // // Push to queue without immediate requeue
// //         // if (!frame.dataVec.empty() && frameQueue_.size() < MAX_QUEUE_SIZE) {
// //         //     frameQueue_.push(frame);
// //         // }
// //         if (!frame.dataVec.empty()) {
// //         if (frameQueue_.size() >= MAX_QUEUE_SIZE) {
// //             frameQueue_.pop(frame); // Discard oldest frame
// //         }
// //         frameQueue_.push(frame);
// //         }
// //     }
// //     catch(const std::exception& e){ 
// //         reportError("Algorithm error: " + std::string(e.what()));
// //     }
// //     }
    
// // }
// // // ---------------------------------------------------------------------
// // // Thread Routine: SoC Loop
// // // ---------------------------------------------------------------------
// // inline void SystemCaptureFactory::SoCLoop(ISoC* soc)
// // {
// //     while (running_) {
// //         if (paused_) {
// //             std::this_thread::sleep_for(std::chrono::milliseconds(500));
// //             continue;
// //         }
// //         auto perf = soc->getPerformance();
// //         spdlog::info("[SoC] CPU1 {}%@{}MHz, CPU {}C, GPU {}C",
// //                      perf.CPU1_Utilization_Percent, 
// //                      perf.CPU1_Frequency_MHz,
// //                      perf.CPU_Temperature_C, 
// //                      perf.GPU_Temperature_C);

// //         std::this_thread::sleep_for(std::chrono::seconds(1));
// //     }
// //     spdlog::info("[SystemCaptureFactory] SoC monitoring loop ended.");
// // }


// // // ---------------------------------------------------------------------
// // // Modified algorithmLoop with buffer release
// // // ---------------------------------------------------------------------
// // inline void SystemCaptureFactory::algorithmLoop(IAlgorithm* alg)
// // {
// //     while (running_) {
// //         FrameData frame;
// //         if (!frameQueue_.pop(frame)) {
// //             std::this_thread::sleep_for(std::chrono::milliseconds(5));
// //             continue;
// //         }

// //         // Process frame
// //         bool success = alg->processFrame(frame);
        
// //         // Release buffer after processing
// //         {
// //             std::lock_guard<std::mutex> lock(bufferMutex_);
// //             activeBuffers_.erase(frame.bufferIndex);
// //         }
        
// //         // Requeue the specific buffer
// //         if (auto cam = getCamera(0)) {
// //             cam->queueBuffer(frame.bufferIndex);
// //         }
// //     }
// // }
// // ---------------------------------------------------------------------
// // Thread Routine: Algorithm Loop
// // ---------------------------------------------------------------------

// // inline void SystemCaptureFactory::algorithmLoop(IAlgorithm* alg)
// // {
// //     while (running_) {
// //         if (paused_) {
// //             std::this_thread::sleep_for(std::chrono::milliseconds(100));
// //             continue;
// //         }

// //         FrameData frame;
// //         if (!frameQueue_.pop(frame)) {
// //             std::this_thread::sleep_for(std::chrono::milliseconds(5));
// //             continue;
// //         }

// //         bool success = alg->processFrame(frame);

// //         {
// //         std::lock_guard<std::mutex> lock(bufferMutex_);
// //         activeBuffers_.erase(frame.bufferIndex); // Release buffer
// //         }
// //         cameras_[0]->queueBuffer(); // Only requeue after processing


// //         if (!success) {
// //             spdlog::warn("[SystemCaptureFactory] Algorithm failed to process frame.");
// //         } else if (display_) {
// //             auto* procPtr = alg->getProcessedBuffer();
// //             if (procPtr) {
// //                 display_->updateProcessedFrame(procPtr, frame.width, frame.height);
// //                 //display_->updateProcessedFrame(procPtr,cameraConfig_.width,cameraConfig_.height);

// //             }
// //         }
// //     }
// //     spdlog::info("[SystemCaptureFactory] Algorithm loop ended.");
// // }


// // ---------------------------------------------------------------------
// // Thread Routine: Display Loop
// // ---------------------------------------------------------------------
// inline void SystemCaptureFactory::displayLoop()
// {
//     while (running_) {
//         if (paused_) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(50));
//             continue;
//         }
//         if (display_) {
//             display_->renderAndPollEvents();
//         } else {
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//         }
//     }
//     spdlog::info("[SystemCaptureFactory] Display loop ended.");
// }


// // // // ---------------------------------------------------------------------

// // #pragma once

// // #include "DataConcrete.h"
// // #include "AlgorithmConcrete.h"
// // #include "SdlDisplayConcrete.h"
// // #include <thread>
// // #include <queue>
// // #include <mutex>
// // #include <condition_variable>
// // #include <atomic>

// // class SystemCaptureFactory {
// // public:
// //     SystemCaptureFactory() : running_(false) {}

// //     ~SystemCaptureFactory() {
// //         stop();
// //     }

// //     void start() {
// //         running_ = true;

// //         // Initialize components
// //         camera_.openDevice("/dev/video0");
// //         camera_.configure({640, 480,false, PixelFormat::YUYV});
// //         camera_.startStreaming();

// //         display_.configure({640, 480, false,"Camera Display"});
// //         display_.initializeDisplay(640, 480);

// //         algorithm_.configure({1}); // Configure algorithm with concurrency level

// //         // Start threads
// //         captureThread_ = std::thread(&SystemCaptureFactory::captureLoop, this);
// //         algorithmThread_ = std::thread(&SystemCaptureFactory::algorithmLoop, this);
// //         displayThread_ = std::thread(&SystemCaptureFactory::displayLoop, this);
// //     }

// //     void stop() {
// //         running_ = false;

// //         // Join threads
// //         if (captureThread_.joinable()) captureThread_.join();
// //         if (algorithmThread_.joinable()) algorithmThread_.join();
// //         if (displayThread_.joinable()) displayThread_.join();

// //         // Stop components
// //         camera_.stopStreaming();
// //         display_.closeDisplay();
// //     }

// // private:
// //     void captureLoop() {
// //         while (running_) {
// //             void* dataPtr = nullptr;
// //             size_t sizeBytes = 0;

// //             if (camera_.dequeFrame(dataPtr, sizeBytes, 9)) {
// //                 FrameData frame{dataPtr, sizeBytes, 640, 480};

// //                 // Enqueue frame for processing
// //                 {
// //                     std::lock_guard<std::mutex> lock(queueMutex_);
// //                     frameQueue_.push(frame);
// //                 }
// //                 queueCondVar_.notify_one();
// //             }

// //             // Simulate frame rate (e.g., 30 FPS)
// //             std::this_thread::sleep_for(std::chrono::milliseconds(33));
// //         }
// //     }

// //     void algorithmLoop() {
// //         while (running_) {
// //             FrameData frame;

// //             // Dequeue frame for processing
// //             {
// //                 std::unique_lock<std::mutex> lock(queueMutex_);
// //                 queueCondVar_.wait(lock, [this] { return !frameQueue_.empty() || !running_; });

// //                 if (!running_) break;

// //                 frame = frameQueue_.front();
// //                 frameQueue_.pop();
// //             }

// //             // Process frame
// //             algorithm_.processFrame(frame);

// //             // Update display with processed frame
// //             display_.updateProcessedFrame(algorithm_.getProcessedBuffer(), 640, 480);

// //             // Push metrics
// //             algorithm_.pushAlgorithmMetrics();
// //         }
// //     }

// //     void displayLoop() {
// //         while (running_) {
// //             // Render frames and poll events
// //             display_.renderAndPollEvents();

// //             // Simulate display refresh rate (e.g., 60 FPS)
// //             std::this_thread::sleep_for(std::chrono::milliseconds(16));
// //         }
// //     }

// // private:
// //     DataConcrete camera_;
// //     AlgorithmConcrete algorithm_;
// //     SdlDisplayConcrete display_;

// //     std::thread captureThread_;
// //     std::thread algorithmThread_;
// //     std::thread displayThread_;

// //     std::queue<FrameData> frameQueue_;
// //     std::mutex queueMutex_;
// //     std::condition_variable queueCondVar_;

// //     std::atomic<bool> running_;
// // };