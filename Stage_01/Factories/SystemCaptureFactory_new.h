//SystemCaptureFactory_new.h

#ifndef SYSTEMCAPTUREFACTORY_NEW_H
#define SYSTEMCAPTUREFACTORY_NEW_H

#include <atomic>
#include <memory>
#include <mutex>
#include <functional>
#include <chrono>
#include <thread>
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
#include "../Concretes/SoCConcrete_new.h"

//-----------------------------------------------------
// Unified Selection Structure for all subsystems
//-----------------------------------------------------
struct CaptureComponentsSelection {
    bool cameraEnabled = true;
    bool algorithmEnabled = false;
    bool soCEnabled = false;
    bool displayEnabled = false;
};

//-----------------------------------------------------
// SystemCaptureFactory Class
//-----------------------------------------------------
class SystemCaptureFactory : public ISystemProfiling {
public:
    SystemCaptureFactory(std::shared_ptr<ISoC> soc,
                         std::shared_ptr<IAlgorithm> algo,
                         std::shared_ptr<IData> camera,
                         std::shared_ptr<SharedQueue<FrameData>> algoQueue,
                         std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue,
                         std::shared_ptr<SharedQueue<FrameData>> processedQueue)
          : running_(false)
        , paused_(false)
        , soc_(std::move(soc))
        , algo_(std::move(algo))
        , camera_(std::move(camera))
        , algoQueue_(std::move(algoQueue))
        , displayOrigQueue_(std::move(displayOrigQueue))
        , processedQueue_(std::move(processedQueue))
    {
        if (!camera_) {
            throw std::invalid_argument("[SystemCaptureFactory] Camera pointer is null!");
        }
    }

    ~SystemCaptureFactory() override;

    /**
     * @brief Initializes all subsystems by calling modular initialization methods.
     *
     * @param cameraConfig Configuration for the camera.
     * @param algoConfig Configuration for the algorithm.
     * @param selection Unified selection for all components.
     * @param socConfig Optional SoC configuration.
     * @param dispConfig Optional Display configuration.
     */
    void initializeCapture(const CameraConfig& cameraConfig,
                           const AlgorithmConfig& algoConfig,
                           const CaptureComponentsSelection& selection,
                           const SoCConfig* socConfig = nullptr,
                           const DisplayConfig* dispConfig = nullptr)
    {
        try {
            spdlog::info("[SystemCaptureFactory] Initializing capture system...");

            // Modular initialization for each subsystem.
            if (!initializeCamera(cameraConfig, selection.cameraEnabled))
                throw std::runtime_error("Camera initialization failed.");

            if (!initializeAlgorithm(algoConfig, selection.algorithmEnabled))
                throw std::runtime_error("Algorithm initialization failed.");

            if (!initializeSoC(socConfig, selection.soCEnabled))
                throw std::runtime_error("SoC initialization failed.");

            if (dispConfig && selection.displayEnabled) {
                if (!initializeDisplay(*dispConfig, selection.displayEnabled))
                    throw std::runtime_error("Display initialization failed.");
            }

            running_ = true;
            paused_  = false;

            // Start threads for enabled subsystems.
            if (selection.soCEnabled && soc_) {
                spdlog::info("[SystemCaptureFactory] Starting SoC monitor thread.");
                threadManager_.addThread("SoCMonitor", std::thread([this]() {
                    try { SoCLoop(soc_.get()); }
                    catch (const std::exception& e) { reportError("SoCMonitor error: " + std::string(e.what())); }
                }));
            }
            if (selection.cameraEnabled) {
                spdlog::info("[SystemCaptureFactory] Starting camera capture thread.");
                threadManager_.addThread("CameraCapture", std::thread([this]() {
                    try { captureLoop(camera_.get()); }
                    catch (const std::exception& e) { reportError("CameraCapture error: " + std::string(e.what())); }
                }));
            }
            if (selection.algorithmEnabled && algo_) {
                spdlog::info("[SystemCaptureFactory] Starting algorithm processing thread.");
                threadManager_.addThread("AlgorithmProcessing", std::thread([this]() {
                    try { algorithmLoop(); }
                    catch (const std::exception& e) { reportError("AlgorithmProcessing error: " + std::string(e.what())); }
                }));
            }
            if (selection.displayEnabled && display_) {
                spdlog::info("[SystemCaptureFactory] Starting display render thread.");
                threadManager_.addThread("DisplayRender", std::thread([this]() {
                    try {
                        while (running_.load()) {
                            display_->renderAndPollEvents();
                            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
                        }
                    } catch (const std::exception& e) {
                        reportError("DisplayRender error: " + std::string(e.what()));
                    }
                }));
            }
            spdlog::info("[SystemCaptureFactory] Initialization complete.");
        } catch (const std::exception& e) {
            reportError(e.what());
            stopCapture();
            throw;
        }
    }

    void stopCapture();
    void pauseAll();
    void resumeAll();

    void setDisplay(std::shared_ptr<IDisplay> display);
    std::shared_ptr<IDisplay> getDisplay() const;

    // Delegated configuration.
    void configureCamera(const CameraConfig& cfg) { if (camera_) camera_->configure(cfg); }
    void configureAlgorithm(const AlgorithmConfig& cfg){ if (algo_) algo_->configure(cfg); }
    void configureSoC(const SoCConfig& cfg) { if (soc_) soc_->configure(cfg); }
    void configureDisplay(const DisplayConfig& cfg) { if (display_) display_->configure(cfg); }
    
    // Profiling / Metrics.
    JetsonNanoInfo getSoCMetrics() const override;
    std::tuple<double, double> getAlgorithmMetrics() const override;
    std::tuple<double, int> getCameraMetrics(int cameraIndex) const override;

    std::shared_ptr<ISoC>       getSoC() const { return soc_; }
    std::shared_ptr<IData>      getCamera() const { return camera_; }
    std::shared_ptr<IAlgorithm> getAlgorithm() const { return algo_; }

    bool isRunning() const { return running_.load(); }

private:
    // Modular initialization methods.
    bool initializeCamera(const CameraConfig& cameraConfig, bool enableCamera);
    bool initializeAlgorithm(const AlgorithmConfig& algoConfig, bool enableAlgorithm);
    bool initializeSoC(const SoCConfig* socConfig, bool enableSoC);
    bool initializeDisplay(const DisplayConfig& dispConfig, bool enableDisplay);

    void captureLoop(IData* camera);
    void algorithmLoop();
    void SoCLoop(ISoC* soc);

    void reportError(const std::string& msg);

    std::atomic<bool> running_;
    std::atomic<bool> paused_;

    std::shared_ptr<ISoC>       soc_;
    std::shared_ptr<IAlgorithm> algo_;
    std::shared_ptr<IData>      camera_;
    std::shared_ptr<IDisplay>   display_;

    ThreadManager threadManager_;

    // Shared queues.
    std::shared_ptr<SharedQueue<FrameData>> algoQueue_;
    std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue_;
    std::shared_ptr<SharedQueue<FrameData>> processedQueue_;
};

//////////////////// Implementation ////////////////////

inline SystemCaptureFactory::~SystemCaptureFactory() {
    stopCapture();
}

inline void SystemCaptureFactory::stopCapture() {
    if (!running_) return;
    running_ = false;
    if (camera_) {
        camera_->stopStreaming();
    }
    threadManager_.joinAll();
    spdlog::info("[SystemCaptureFactory] Capture system stopped.");
}

inline bool SystemCaptureFactory::initializeCamera(const CameraConfig& cameraConfig, bool enableCamera) {
    spdlog::info("[SystemCaptureFactory] Initializing Camera...");
    if (!camera_->openDevice("/dev/video0")) {
        spdlog::error("[SystemCaptureFactory] Failed to open /dev/video0");
        return false;
    }
    if (!camera_->configure(cameraConfig)) {
        spdlog::error("[SystemCaptureFactory] Failed to configure camera");
        return false;
    }
    if (enableCamera) {
        if (!camera_->startStreaming()) {
            spdlog::error("[SystemCaptureFactory] Failed to start camera streaming");
            return false;
        }
        spdlog::info("[SystemCaptureFactory] Camera streaming started.");
    } else {
        spdlog::info("[SystemCaptureFactory] Camera disabled by selection.");
    }
    return true;
}

inline bool SystemCaptureFactory::initializeAlgorithm(const AlgorithmConfig& algoConfig, bool enableAlgorithm) {
    spdlog::info("[SystemCaptureFactory] Initializing Algorithm...");
    if (algo_ && enableAlgorithm) {
        if (!algo_->configure(algoConfig)) {
            spdlog::error("[SystemCaptureFactory] Algorithm configuration failed");
            return false;
        }
        spdlog::info("[SystemCaptureFactory] Algorithm configured.");
    } else {
        spdlog::info("[SystemCaptureFactory] Algorithm disabled by selection or not available.");
    }
    return true;
}

inline bool SystemCaptureFactory::initializeSoC(const SoCConfig* socConfig, bool enableSoC) {
    spdlog::info("[SystemCaptureFactory] Initializing SoC...");
    if (soc_ && enableSoC) {
        if (socConfig) {
            soc_->configure(*socConfig);
        }
        soc_->initializeSoC(); // Assuming initializeSoC() does not return a bool.
        spdlog::info("[SystemCaptureFactory] SoC initialized.");
    } else {
        spdlog::info("[SystemCaptureFactory] SoC disabled by selection or not available.");
    }
    return true;
}

inline bool SystemCaptureFactory::initializeDisplay(const DisplayConfig& dispConfig, bool enableDisplay) {
    spdlog::info("[SystemCaptureFactory] Initializing Display...");
    if (display_ && enableDisplay) {
        display_->configure(dispConfig);
        if (!display_->initializeDisplay(dispConfig.width, dispConfig.height)) {
            spdlog::error("[SystemCaptureFactory] Failed to initialize display!");
            return false;
        }
        spdlog::info("[SystemCaptureFactory] Display initialized.");
    } else {
        spdlog::info("[SystemCaptureFactory] Display disabled by selection or not available.");
    }
    return true;
}


inline void SystemCaptureFactory::captureLoop(IData* camera) {
    try {
        while (running_.load()) {
            spdlog::debug("[CaptureLoop] Starting capture cycle.");
            void* dataPtr = nullptr;
            size_t sizeBytes = 0;
            size_t bufferIndex = 0;
            if (!camera->dequeFrame(dataPtr, sizeBytes, bufferIndex)) {
                spdlog::debug("[CaptureLoop] No frame available; sleeping for 16 ms.");
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
                continue;
            }
            spdlog::debug("[CaptureLoop] Successfully dequeued buffer {} (size={}).", bufferIndex, sizeBytes);
            // Automatic requeue is assumed inside dequeFrame().
        }
    } catch (const std::exception& e) {
        reportError(std::string("Capture loop error: ") + e.what());
    }
}

inline void SystemCaptureFactory::algorithmLoop() {
    try {
        while (running_.load()) {
            FrameData frame;
            if (!algoQueue_->pop(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            spdlog::debug("[AlgorithmLoop] Popped frame from AlgoQueue (current size: {}).", algoQueue_->size());
            processedQueue_->push(frame);
        }
    } catch (const std::exception& e) {
        reportError(std::string("Algorithm loop error: ") + e.what());
    }
}

inline void SystemCaptureFactory::SoCLoop(ISoC* soc) {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (soc) {
            auto info = soc->getPerformance();
            spdlog::debug("[SoCLoop] SoC performance updated.");
        }
    }
}

inline void SystemCaptureFactory::pauseAll() {
    paused_ = true;
    // Additional pause logic if necessary.
}

inline void SystemCaptureFactory::resumeAll() {
    paused_ = false;
    // Additional resume logic if necessary.
}

inline void SystemCaptureFactory::reportError(const std::string& msg) {
    spdlog::error("[SystemCaptureFactory] {}", msg);
}

inline void SystemCaptureFactory::setDisplay(std::shared_ptr<IDisplay> display) {
    display_ = display;
}

inline std::shared_ptr<IDisplay> SystemCaptureFactory::getDisplay() const {
    return display_;
}

inline JetsonNanoInfo SystemCaptureFactory::getSoCMetrics() const {
    return soc_ ? soc_->getPerformance() : JetsonNanoInfo{};
}

inline std::tuple<double, double> SystemCaptureFactory::getAlgorithmMetrics() const {
    if (!algo_) return {0.0, 0.0};
    return algo_->getAlgorithmMetrics();
}

inline std::tuple<double, int> SystemCaptureFactory::getCameraMetrics(int) const {
    if (!camera_) {
        spdlog::warn("[SystemCaptureFactory] No camera instance available.");
        return {0.0, 0};
    }
    return {camera_->getLastFPS(), camera_->getQueueSize()};
}

#endif // SYSTEMCAPTUREFACTORY_NEW_H




//===========================================================================
// // SystemCaptureFactory_new.h

// // (Stage 2 Factory: Capture & Frame Processing Pipeline)

//=============================================0 working !!
// #ifndef SYSTEMCAPTUREFACTORY_NEW_H
// #define SYSTEMCAPTUREFACTORY_NEW_H

// #include <atomic>
// #include <memory>
// #include <mutex>
// #include <functional>
// #include <spdlog/spdlog.h>
// #include <SDL2/SDL.h>
// #include <fmt/format.h>

// // Interfaces
// #include "../Interfaces/IData.h"
// #include "../Interfaces/IAlgorithm.h"
// #include "../Interfaces/ISoC.h"
// #include "../Interfaces/IDisplay.h"
// #include "../Interfaces/ISystemProfiling.h"

// // Shared Structures
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/AlgorithmConfig.h"
// #include "../SharedStructures/SoCConfig.h"
// #include "../SharedStructures/DisplayConfig.h"
// #include "../SharedStructures/SharedQueue.h"

// // Thread Management
// #include "../SharedStructures/ThreadManager.h"

// // Concrete Implementations
// #include "../Concretes/DataConcrete_new.h"
// #include "../Concretes/AlgorithmConcrete_new.h"
// #include "../Concretes/SdlDisplayConcrete_new.h"
// #include "../Concretes/SoCConcrete_new.h"

// /**
//  * @struct CaptureSelection
//  * @brief Determines which system components to enable (camera, algorithm, SoC).
//  */
// struct CaptureSelection {
//     bool enableCamera    = true;  
//     bool enableAlgorithm = false; 
//     bool enableSoC       = false; 
// };

// /**
//  * @class SystemCaptureFactory
//  * @brief Manages the end-to-end capture (camera), processing (algorithm), and display pipeline.
//  */
// class SystemCaptureFactory : public ISystemProfiling {
// public:
//     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                          std::shared_ptr<IAlgorithm> algo,
//                          std::shared_ptr<IData> camera,
//                          std::shared_ptr<SharedQueue<FrameData>> algoQueue,
//                          std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue,
//                          std::shared_ptr<SharedQueue<FrameData>> processedQueue)
//           : running_(false)
//         , paused_(false) 
//         , soc_(std::move(soc))
//         , algo_(std::move(algo))
//         , camera_(std::move(camera))
//         , algoQueue_(std::move(algoQueue))
//         , displayOrigQueue_(std::move(displayOrigQueue))
//         , processedQueue_(std::move(processedQueue))
//     {
//         if (!camera_) {
//             throw std::invalid_argument("[SystemCaptureFactory] Camera pointer is null!");
//         }
//     }

//     ~SystemCaptureFactory() override;

//     // Lifecycle
//     void initializeCapture(const CameraConfig& cameraConfig, 
//                            const AlgorithmConfig& algoConfig, 
//                            const CaptureSelection& captureSelection);
//     void stopCapture();
//     void pauseAll();
//     void resumeAll();

//     // Display management
//     void setDisplay(std::shared_ptr<IDisplay> display);
//     std::shared_ptr<IDisplay> getDisplay() const;

//     // Configuration methods
//     void configureCamera(const CameraConfig& cfg) { if (camera_) camera_->configure(cfg); }
//     void configureAlgorithm(const AlgorithmConfig& cfg){ if (algo_) algo_->configure(cfg); }
//     void configureSoC(const SoCConfig& cfg) { if (soc_) soc_->configure(cfg); }
//     void configureDisplay(const DisplayConfig& cfg) { if (display_) display_->configure(cfg); }
    
//     // Profiling / Metrics
//     JetsonNanoInfo getSoCMetrics() const override;
//     std::tuple<double, double> getAlgorithmMetrics() const override;
//     std::tuple<double, int> getCameraMetrics(int cameraIndex) const override;

//     // Get components
//     std::shared_ptr<ISoC>       getSoC()       const { return soc_; }
//     std::shared_ptr<IData>      getCamera()    const { return camera_; }
//     std::shared_ptr<IAlgorithm> getAlgorithm() const { return algo_; }

//     // Accessor for running state.
//     bool isRunning() const { return running_.load(); }

// private:
//     void captureLoop(IData* camera);
//     void algorithmLoop();
//     void SoCLoop(ISoC* soc);

//     void reportError(const std::string& msg);

//     std::atomic<bool> running_;
//     std::atomic<bool> paused_;

//     std::shared_ptr<ISoC>       soc_;
//     std::shared_ptr<IAlgorithm> algo_;
//     std::shared_ptr<IData>      camera_;
//     std::shared_ptr<IDisplay>   display_;

//     ThreadManager threadManager_;

//     // Shared queues injected from the factory.
//     std::shared_ptr<SharedQueue<FrameData>> algoQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> processedQueue_;
// };

// //////////////////// Implementation ////////////////////

// inline SystemCaptureFactory::~SystemCaptureFactory() {
//     stopCapture();
// }

// inline void SystemCaptureFactory::initializeCapture(const CameraConfig& cameraConfig,
//                                                     const AlgorithmConfig& algoConfig,
//                                                     const CaptureSelection& captureSelection)
// {
//     try {
//         spdlog::info("[SystemCaptureFactory] Initializing capture system...");

//         if (!camera_->openDevice("/dev/video0")) {
//             throw std::runtime_error("[SystemCaptureFactory] Failed to open /dev/video0");
//         }
//         if (!camera_->configure(cameraConfig)) {
//             throw std::runtime_error("[SystemCaptureFactory] Failed to configure camera");
//         }
//         if (captureSelection.enableCamera) {
//             camera_->startStreaming();
//             spdlog::info("[SystemCaptureFactory] Camera streaming started.");
//         }

//         if (algo_ && captureSelection.enableAlgorithm) {
//             algo_->configure(algoConfig);
//             spdlog::info("[SystemCaptureFactory] Algorithm configured.");
//         }
//         if (soc_ && captureSelection.enableSoC) {
//             soc_->initializeSoC();
//             spdlog::info("[SystemCaptureFactory] SoC initialized.");
//         }

//         if (display_) {
//             DisplayConfig dispCfg;// = { cameraConfig.width, cameraConfig.height, false, "Framework Display" };
//             display_->configure(dispCfg);
//             if (!display_->initializeDisplay(dispCfg.width, dispCfg.height)) {
//                 throw std::runtime_error("Failed to initialize display!");
//             }
//             spdlog::info("[SystemCaptureFactory] Display initialized.");
//         }

//         running_ = true;
//         paused_  = false;

//         // Start SoC monitor thread if enabled.
//         if (captureSelection.enableSoC) {
//             spdlog::info("[SystemCaptureFactory] Enabling SoC monitoring.");
//             threadManager_.addThread("SoCMonitor", std::thread([this]() {
//                 SoCLoop(soc_.get());
//             }));
//         }

//         // Start the camera capture thread.
//         if (captureSelection.enableCamera) {
//             spdlog::info("[SystemCaptureFactory] Enabling camera capture.");
//             spdlog::info("[SystemCaptureFactory] initializeCapture Enable Camera.");
//             threadManager_.addThread("CameraCapture", std::thread([this]() {
//                 captureLoop(camera_.get());
//             }));
//         }

//         // Start algorithm processing thread if enabled.
//         if (captureSelection.enableAlgorithm) {
//             spdlog::info("[SystemCaptureFactory] Enabling algorithm processing.");
//             threadManager_.addThread("AlgorithmProcessing", std::thread([this]() {
//                 algorithmLoop();
//             }));
//         }

//         // Start display render thread.
//         if (display_) {
//             spdlog::info("[SystemCaptureFactory] Starting display render thread.");
//             threadManager_.addThread("DisplayRender", std::thread([this]() {
//                 while (running_.load()) {
//                     display_->renderAndPollEvents();
//                     std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
//                 }
//             }));
//         }

//         spdlog::info("[SystemCaptureFactory] Initialization complete.");
//     } catch (const std::exception& e) {
//         reportError(e.what());
//         stopCapture();
//         throw;
//     }
// }

// inline void SystemCaptureFactory::stopCapture() {
//     if (!running_) return;
//     running_ = false;

//     if (camera_) {
//         camera_->stopStreaming();
//     }
//     threadManager_.joinAll();
//     spdlog::info("[SystemCaptureFactory] Capture system stopped.");
// }

// inline void SystemCaptureFactory::captureLoop(IData* camera) {
//      try {
//     while (running_.load()) {
//         spdlog::debug("[CaptureLoop] Starting capture cycle.");
//         void* dataPtr = nullptr;
//         size_t sizeBytes = 0;
//         size_t bufferIndex = 0;
//         // If dequeFrame() fails, immediately sleep and continue.
//         if (!camera->dequeFrame(dataPtr, sizeBytes, bufferIndex)) {
//             spdlog::debug("[CaptureLoop] No frame available; sleeping for 16 ms.");
//             std::this_thread::sleep_for(std::chrono::milliseconds(16));
//             continue;
//         }
//         spdlog::debug("[CaptureLoop] Successfully dequeued buffer {} (size={}).", bufferIndex, sizeBytes);
//         // Note: Do not call queueBuffer() here if dequeFrame() already handles automatic requeue.
//     }
//     } catch (const std::exception& e) {
//         reportError(std::string("Capture loop error: ") + e.what());
//     }
// }

// inline void SystemCaptureFactory::algorithmLoop() {
//     try {
//     while (running_.load()) {
//         FrameData frame;
//         if (!algoQueue_->pop(frame)) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(5));
//             continue;
//         }
//         spdlog::debug("[AlgorithmLoop] Popped frame from AlgoQueue (current size: {}).", algoQueue_->size());
//         // For this example, simply push the unmodified frame to the processed queue.
//         processedQueue_->push(frame);
//     }
//     } catch (const std::exception& e) {
//         reportError(std::string("Algorithm loop error: ") + e.what());
//     }
// }

// inline void SystemCaptureFactory::SoCLoop(ISoC* soc) {
//     while (running_.load()) {
//         std::this_thread::sleep_for(std::chrono::seconds(1));
//         if (soc) {
//             auto info = soc->getPerformance();
//             spdlog::debug("[SoCLoop] SoC performance updated.");
//         }
//     }
// }

// inline void SystemCaptureFactory::pauseAll() {
//     paused_ = true;
//     // Additional pause logic if necessary.
// }

// inline void SystemCaptureFactory::resumeAll() {
//     paused_ = false;
//     // Additional resume logic if necessary.
// }

// inline void SystemCaptureFactory::reportError(const std::string& msg) {
//     spdlog::error("[SystemCaptureFactory] {}", msg);
// }

// inline void SystemCaptureFactory::setDisplay(std::shared_ptr<IDisplay> display) {
//     display_ = display;
// }

// inline std::shared_ptr<IDisplay> SystemCaptureFactory::getDisplay() const {
//     return display_;
// }

// inline JetsonNanoInfo SystemCaptureFactory::getSoCMetrics() const {
//     return soc_ ? soc_->getPerformance() : JetsonNanoInfo{};
// }

// inline std::tuple<double, double> SystemCaptureFactory::getAlgorithmMetrics() const {
//     if (!algo_) return {0.0, 0.0};
//     return algo_->getAlgorithmMetrics();
// }

// inline std::tuple<double, int> SystemCaptureFactory::getCameraMetrics(int) const {
//     if (!camera_) {
//         spdlog::warn("[SystemCaptureFactory] No camera instance available.");
//         return {0.0, 0};
//     }
//     return {camera_->getLastFPS(), camera_->getQueueSize()};
// }

// #endif // SYSTEMCAPTUREFACTORY_NEW_H


// =================================================================
// // SystemCaptureFactory_new.h

// // (Stage 2 Factory: Capture & Frame Processing Pipeline)


// #ifndef SYSTEMCAPTUREFACTORY_NEW_H
// #define SYSTEMCAPTUREFACTORY_NEW_H

// #include <atomic>
// #include <memory>
// #include <mutex>
// #include <functional>
// #include <spdlog/spdlog.h>
// #include <SDL2/SDL.h>
// #include <fmt/format.h>

// // Interfaces
// #include "../Interfaces/IData.h"
// #include "../Interfaces/IAlgorithm.h"
// #include "../Interfaces/ISoC.h"
// #include "../Interfaces/IDisplay.h"
// #include "../Interfaces/ISystemProfiling.h"

// // Shared Structures
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/AlgorithmConfig.h"
// #include "../SharedStructures/SoCConfig.h"
// #include "../SharedStructures/DisplayConfig.h"
// #include "../SharedStructures/SharedQueue.h"

// // Thread Management
// #include "../SharedStructures/ThreadManager.h"

// // Concrete Implementations
// #include "../Concretes/DataConcrete_new.h"
// #include "../Concretes/AlgorithmConcrete_new.h"
// #include "../Concretes/SdlDisplayConcrete_new.h"
// #include "../Concretes/SoCConcrete_new.h"

// /**
//  * @struct CaptureSelection
//  * @brief Determines which system components to enable (camera, algorithm, SoC).
//  */
// struct CaptureSelection {
//     bool enableCamera    = true;  
//     bool enableAlgorithm = true; 
//     bool enableSoC       = false; 
// };

// /**
//  * @class SystemCaptureFactory
//  * @brief Manages the end-to-end capture (camera), processing (algorithm), and display pipeline.
//  *
//  * - If camera is enabled, frames are captured and pushed to the shared queues.
//  * - If algorithm is enabled, frames are popped from algoQueue and processed, then pushed to processedQueue.
//  * - If SoC is enabled, SoC metrics are updated in a separate thread.
//  * - The display can be set from outside and will read from the shared queues.
//  */
// class SystemCaptureFactory : public ISystemProfiling {
// public:
//     SystemCaptureFactory(std::shared_ptr<ISoC> soc,
//                          std::shared_ptr<IAlgorithm> algo,
//                          std::shared_ptr<IData> camera,
//                          std::shared_ptr<SharedQueue<FrameData>> algoQueue,
//                          std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue,
//                          std::shared_ptr<SharedQueue<FrameData>> processedQueue)
//           : running_(false)
//         , paused_(false) 
//         , soc_(std::move(soc))
//         , algo_(std::move(algo))
//         , camera_(std::move(camera))
//         , algoQueue_(std::move(algoQueue))
//         , displayOrigQueue_(std::move(displayOrigQueue))
//         , processedQueue_(std::move(processedQueue))
// {
//     if (!camera_) {
//         throw std::invalid_argument("[SystemCaptureFactory] Camera pointer is null!");
//     }
// }


//     ~SystemCaptureFactory() override;

//     // Lifecycle
//     void initializeCapture(const CameraConfig&    cameraConfig, 
//                            const AlgorithmConfig& algoConfig, 
//                            const CaptureSelection& captureSelection);
//     void stopCapture();
//     void pauseAll();
//     void resumeAll();

//     // Display management
//     void setDisplay(std::shared_ptr<IDisplay> display);
//     std::shared_ptr<IDisplay> getDisplay() const;

//     // Configuration methods
//     void configureCamera(const CameraConfig& cfg)      { if (camera_) camera_->configure(cfg); }
//     void configureAlgorithm(const AlgorithmConfig& cfg){ if (algo_)   algo_->configure(cfg);   }
//     void configureSoC(const SoCConfig& cfg)            { if (soc_)    soc_->configure(cfg);    }
//     void configureDisplay(const DisplayConfig& cfg)    { if (display_)display_->configure(cfg);}
    
//     // Profiling / Metrics
//     JetsonNanoInfo getSoCMetrics() const override;
//     std::tuple<double, double> getAlgorithmMetrics() const override;
//     std::tuple<double, int>    getCameraMetrics(int cameraIndex) const override;

//     // Get components
//     std::shared_ptr<ISoC>       getSoC()       const { return soc_;    }
//     std::shared_ptr<IData>      getCamera()    const { return camera_; }
//     std::shared_ptr<IAlgorithm> getAlgorithm() const { return algo_;   }

//     // Add this accessor:
//     bool isRunning() const { return running_.load(); }

// private:
//     void captureLoop(IData* camera);
//     void algorithmLoop();
//     void SoCLoop(ISoC* soc);

//     void reportError(const std::string& msg);

//     std::atomic<bool> running_;
//     std::atomic<bool> paused_;

//     std::shared_ptr<ISoC>       soc_;
//     std::shared_ptr<IAlgorithm> algo_;
//     std::shared_ptr<IData>      camera_;
//     std::shared_ptr<IDisplay>   display_;

//     ThreadManager threadManager_;

//     // Per-instance queues (these could also be passed in, but here we assume local creation)
//     std::shared_ptr<SharedQueue<FrameData>> algoQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> processedQueue_;
// };

// // ------------------- Implementation -------------------- //

// // inline SystemCaptureFactory::SystemCaptureFactory(std::shared_ptr<ISoC> soc,
// //                          std::shared_ptr<IAlgorithm> algo,
// //                          std::shared_ptr<IData> camera,
// //                          std::shared_ptr<SharedQueue<FrameData>> algoQueue,
// //                          std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue,
// //                          std::shared_ptr<SharedQueue<FrameData>> processedQueue)
// //           : running_(false)
// //         , paused_(false) 
// //         , soc_(std::move(soc))
// //         , algo_(std::move(algo))
// //         , camera_(std::move(camera))
// //         , algoQueue_(std::move(algoQueue))
// //         , displayOrigQueue_(std::move(displayOrigQueue))
// //         , processedQueue_(std::move(processedQueue))
// // {
// //     if (!camera_) {
// //         throw std::invalid_argument("[SystemCaptureFactory] Camera pointer is null!");
// //     }
// // }

// inline SystemCaptureFactory::~SystemCaptureFactory() {
//     stopCapture();
// }

// inline void SystemCaptureFactory::initializeCapture(const CameraConfig&    cameraConfig,
//                                                     const AlgorithmConfig& algoConfig,
//                                                     const CaptureSelection& captureSelection)
// {
//     try {
//         spdlog::info("[SystemCaptureFactory] Initializing capture system...");

//         if (!camera_->openDevice("/dev/video0")) {
//             throw std::runtime_error("[SystemCaptureFactory] Failed to open /dev/video0");
//         }
//         if (!camera_->configure(cameraConfig)) {
//             throw std::runtime_error("[SystemCaptureFactory] Failed to configure camera");
//         }
//         if (captureSelection.enableCamera) {
//             camera_->startStreaming();
//         }

//         if (algo_ && captureSelection.enableAlgorithm) {
//             algo_->configure(algoConfig);
//         }
//         if (soc_ && captureSelection.enableSoC) {
//             soc_->initializeSoC();
//         }

//         // Display configuration
//         // Configure display if available
//         // if (display_) {

//         //     DisplayConfig dispCfg = {cameraConfig.width, cameraConfig.height, false, "Framework Display"};
//         //     display_->configure(dispCfg);
//         //     display_->initializeDisplay(dispCfg.width, dispCfg.height);
//         // }

//         // When configuring the display:
//         if (display_) {
//             DisplayConfig dispCfg = {
//                 cameraConfig.width, 
//                 cameraConfig.height, 
//                 false, 
//                 "Framework Display"
//             };
//             display_->configure(dispCfg);
//             if (!display_->initializeDisplay(dispCfg.width, dispCfg.height)) {
//                 throw std::runtime_error("Failed to initialize display!");
//             }
//         }


//           // Do NOT recreate the queues here; they were injected already.!!!
//         // Remove these lines:
//         // algoQueue_        = std::make_shared<SharedQueue<FrameData>>(5);
//         // displayOrigQueue_ = std::make_shared<SharedQueue<FrameData>>(5);
//         // processedQueue_   = std::make_shared<SharedQueue<FrameData>>(5);


//          // ... start threads, etc.
//         running_ = true;
//         paused_  = false;

//         // Start threads as needed
//         // SoC monitoring
    

//         if (captureSelection.enableSoC) {
//             spdlog::info("[SystemCaptureFactory] initializeCapture Enable SoC.");
//             threadManager_.addThread("SoCMonitor", std::thread([this]() {
//                 SoCLoop(soc_.get());
//             }));
//         }
//         // Camera capture
//         // threadManager_.addThread("CameraCapture", std::thread([this]() {
//         //       captureLoop(camera_.get());
//         //     }));
        


//         if (captureSelection.enableCamera) {
//             spdlog::info("[SystemCaptureFactory] initializeCapture Enable Camera.");
//             threadManager_.addThread("CameraCapture", std::thread([this]() {
//                 captureLoop(camera_.get());
//             }));
//         }
//         // Algorithm processing
//         if (captureSelection.enableAlgorithm) {
//             spdlog::info("[SystemCaptureFactory] initializeCapture:  Enable Algorithm.");
//             threadManager_.addThread("AlgorithmProcessing", std::thread([this]() {
//                 algorithmLoop();
//             }));
//         }

//                 // Inside the initializeCapture method, after initializing the display:
//         if (display_) {
//             spdlog::info("[SystemCaptureFactory] Starting display render thread.");
//             threadManager_.addThread("DisplayRender", std::thread([this]() {
//                 while (running_.load()) {
//                     display_->renderAndPollEvents();
//                     std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
//                 }
//             }));
//         }

//         spdlog::info("[SystemCaptureFactory] Initialization complete.");
//     } catch (const std::exception& e) {
//         reportError(e.what());
//         stopCapture();
//         throw;
//     }
// }

// inline void SystemCaptureFactory::stopCapture() {
//     if (!running_) return;
//     running_ = false;

//     if (camera_) {
//         camera_->stopStreaming();
//     }
//     // Join threads
//     threadManager_.joinAll();
//     spdlog::info("[SystemCaptureFactory] Capture system stopped.");
// }

// // inline void SystemCaptureFactory::captureLoop(IData* camera) {
// //     // This loop simulates pulling frames from camera->dequeFrame(...)
// //     while (running_) {
// //         void* dataPtr = nullptr;
// //         size_t sizeBytes = 0;
// //         size_t bufferIndex = 0;

// //         if (!camera->dequeFrame(dataPtr, sizeBytes, bufferIndex)) {
// //             std::this_thread::sleep_for(std::chrono::milliseconds(10));
// //             continue;
// //         }

// //         // We already push frames from inside DataConcrete’s dequeFrame in the new design,
// //         // but if you need an additional local push, you could do it here:
// //         // e.g. displayOrigQueue_->push(...);
// //          //displayOrigQueue_->push(bufferIndex);


// //         // Re-queue the buffer so it can be used again
// //         camera->queueBuffer(bufferIndex);
// //     }
// // }

// void SystemCaptureFactory::captureLoop(IData* camera) {
//     while (running_) {
//         spdlog::debug("[DataConcrete] captureLoop start ¡¡¡¡¡");
//         void* dataPtr = nullptr;
//         size_t sizeBytes = 0;
//         size_t bufferIndex = 0;

//         // This will do the entire deque->copy->queue->re-queue cycle
//         if (!camera->dequeFrame(dataPtr, sizeBytes, bufferIndex)) {
//             // If no frame, sleep a bit
//             std::this_thread::sleep_for(std::chrono::milliseconds(16));
//         }
//         // Re-queue the buffer so the driver can reuse it
//         camera->queueBuffer(bufferIndex);
//         spdlog::debug("[DataConcrete] captureLoop index={}, size={}", bufferIndex, sizeBytes);

//     }
// }


// inline void SystemCaptureFactory::algorithmLoop() {
//     // If you want to directly read frames from algoQueue_ and further process them here:
//     // This could be a second-level pipeline step
//     while (running_) {
//         FrameData frame;
//         if (!algoQueue_->pop(frame)) {
//             // If queue is stopped or empty
//             std::this_thread::sleep_for(std::chrono::milliseconds(5));
//             continue;
//         }
//         // If needed, do additional processing...
//         // Then push to processedQueue_
//         processedQueue_->push(frame);
//     }
// }

// inline void SystemCaptureFactory::SoCLoop(ISoC* soc) {
//     while (running_) {
//         // For example, read SoC data every second
//         std::this_thread::sleep_for(std::chrono::seconds(1));
//         if (soc) {
//             auto info = soc->getPerformance(); 
//             // Optionally, do something with info...
//         }
//     }
// }

// inline void SystemCaptureFactory::pauseAll() {
//     paused_ = true;
//     // Implementation depends on how you want to pause threads
//     // Possibly set a flag that captureLoop, algorithmLoop, SoCLoop check
// }

// inline void SystemCaptureFactory::resumeAll() {
//     paused_ = false;
//     // Implementation depends on how you want to resume threads
// }

// inline void SystemCaptureFactory::reportError(const std::string& msg) {
//     spdlog::error("[SystemCaptureFactory] {}", msg);
// }

// // Display
// inline void SystemCaptureFactory::setDisplay(std::shared_ptr<IDisplay> display) {
//     display_ = display;
// }
// inline std::shared_ptr<IDisplay> SystemCaptureFactory::getDisplay() const {
//     return display_;
// }

// // Metrics
// inline JetsonNanoInfo SystemCaptureFactory::getSoCMetrics() const {
//     return soc_ ? soc_->getPerformance() : JetsonNanoInfo{};
// }
// inline std::tuple<double, double> SystemCaptureFactory::getAlgorithmMetrics() const {
//     if (!algo_) return {0.0, 0.0};
//     return algo_->getAlgorithmMetrics();
// }
// inline std::tuple<double, int> SystemCaptureFactory::getCameraMetrics(int) const {
//     if (!camera_) {
//         spdlog::warn("[SystemCaptureFactory] No camera instance available.");
//         return {0.0, 0};
//     }
//     return {camera_->getLastFPS(), camera_->getQueueSize()};
// }

// #endif // SYSTEMCAPTUREFACTORY_NEW_H
