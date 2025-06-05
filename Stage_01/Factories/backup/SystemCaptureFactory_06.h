// SystemCaptureFactory.h

#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <SDL2/SDL.h>

#include "../Interfaces/IData.h"
#include "../Interfaces/IAlgorithm.h"
#include "../Interfaces/ISoC.h"
#include "../SharedStructures/FrameData.h"
#include "../SharedStructures/CameraConfig.h"
#include "../SharedStructures/AlgorithmConfig.h"
#include "../SharedStructures/SoCConfig.h"
#include "../SharedStructures/SharedQueue.h"
#include "SystemModellingFactory.h" // So we can use modelingFactory_

#include "../Interfaces/ISystemProfiling.h"

#include <chrono>
#include <pthread.h>
#include <sched.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <SDL2/SDL.h>
#include <spdlog/spdlog.h>

#include "Concretes/SdlDisplayConcrete.h"


class SystemCaptureFactory : public ISystemProfiling {
public:
    /**
     * @brief Constructs a SystemCaptureFactory object.
     * 
     * 
     * This constructor initializes the SystemCaptureFactory object with shared pointers
     * to its dependent components. Dependency injection ensures that the factory is 
     * decoupled from the creation of these components, promoting modularity and testability.
     * 
     *
     * @param soc A shared pointer to an instance of SoCConcrete, representing the System-on-Chip.
     * @param algo A shared pointer to an instance of AlgorithmConcrete, representing the algorithm being used.
     * @param camera0 A shared pointer to an instance of DataConcrete, representing the first camera.
     * @param camera1 A shared pointer to an instance of DataConcrete, representing the second camera.
     * Constructor:
     * The constructor of SystemCaptureFactory takes four std::shared_ptr arguments:
     *  --> soc: A shared pointer to an instance of SoCConcrete, representing the System-on-Chip.
     *  --> algo: A shared pointer to an instance of AlgorithmConcrete, representing the algorithm being used.
     *  --> camera0: A shared pointer to an instance of DataConcrete, representing the first camera.
     *  --> camera1: A shared pointer to an instance of DataConcrete, representing the second camera.
    
    //Using unique_ptr to shared_ptr
    SystemCaptureFactory(std::unique_ptr<ISoC> soc,
                         std::unique_ptr<IAlgorithm> algo,
                         std::unique_ptr<IData> camera0,
                         std::unique_ptr<IData> camera1)

     * 
     */
    SystemCaptureFactory(std::shared_ptr<SoCConcrete> soc,
                         std::shared_ptr<AlgorithmConcrete> algo,
                         std::shared_ptr<DataConcrete> camera0,
                         std::shared_ptr<DataConcrete> camera1)
        // std::move(soc): This uses the std::move function to transfer ownership of the soc shared pointer 
        // from the caller to the SystemCaptureFactory object. This improves efficiency by avoiding unnecessary copying of the shared pointer. 
        // Similar std::move operations are performed for algo, camera0, and camera1.

        // Member Initialization: The member variables soc_, algo_, camera0_, and camera1_ 
        // are initialized with the moved shared pointers. These member variables will be used within the SystemCaptureFactory class 
        // to access and interact with the respective components.
        // Member initializer list:
        // Moves the shared pointers into the respective member variables, ensuring ownership transfer.
        : soc_(std::move(soc)),         // Initialize SoCConcrete dependency
          algo_(std::move(algo)),       // Initialize AlgorithmConcrete dependency
          camera0_(std::move(camera0)), // Initialize Camera 0 (DataConcrete)
          camera1_(std::move(camera1)),  // Initialize Camera 1 (DataConcrete)
          running_(false),
          paused_(false),
          errorCallback_(nullptr)
        {}

    ~SystemCaptureFactory();

    

    /*
    ISystemProfiling.h
    // Get SoC metrics: CPU usage, GPU usage, CPU temp, GPU temp
    virtual std::tuple<double, double, double, double> getSoCMetrics() const = 0;
    */
    
     // Implementation of ISystemProfiling methods
    JetsonNanoInfo getSoCMetrics() const override{
    std::lock_guard<std::mutex> lock(performanceMutex_);
    if (lastPerformanceData_.CPU1_Utilization_Percent == 0 && lastPerformanceData_.GR3D_Frequency_Percent == 0) {
        if (errorCallback_) {
            errorCallback_("[SoCConcrete] No valid performance data available.");
        }
        return JetsonNanoInfo(); // Return a default-initialized structure
    }
    return lastPerformanceData_;
}


    /*
    ISystemProfiling.h  
    // Get camera metrics: FPS and queue size for a specific camera
    virtual std::tuple<double, int> getCameraMetrics(int cameraIndex) const = 0;
  */

       std::tuple<double, int> getCameraMetrics(int cameraIndex) const override {
        auto camera = (cameraIndex == 0) ? camera0_ : camera1_;
        return {camera->getLastFPS(), camera->getQueueSize()};
    }

    /*
    ISystemProfiling.h  
    
    // Get algorithm metrics: FPS and average processing time
    virtual std::tuple<double, double> getAlgorithmMetrics() const = 0;
     */

    std::tuple<double, double> getAlgorithmMetrics() const override{
    if (!algo_) {
        spdlog::error("[AlgorithmConcrete] Algorithm instance is null.");
        return {0.0, 0.0}; // Default metrics for error case
    }
    return {algo_->getFps(), algo_->getAverageProcTime()};
}



   // void initializeSystem();
    void initializeCapture ();
    void stopCapture ();
   // void stopSystem();

    void pauseAll();
    void resumeAll();

    // If you eventually want to set CPU affinity, implement it or remove the parameter name:
    void setThreadAffinity(int /*cpuCore*/);

    void configureCamera(size_t /*index*/, const CameraConfig& /*cfg*/);
    void configureAlgorithm(size_t /*index*/, const AlgorithmConfig& /*cfg*/);
    void configureSoC(const SoCConfig& /*cfg*/);

    void setGlobalErrorCallback(std::function<void(const std::string&)>);
    
private:
    // Thread loops
    void captureLoop(IData* camera);
    void monitorLoop(ISoC* soc);
    void algorithmLoop(IAlgorithm* alg);

    // Display methods
    //void displayLoop(IDisplay* display); // Add a display loop
    void displayLoop(); // Add a display loop

    
private:
    std::atomic<bool> running_;
    std::atomic<bool> paused_;


    // The missing piece: a SystemModelingFactory instance
    SystemModelingFactory modelingFactory_;  

    // Data
/*Using unique_ptr to shared_ptr

    std::vector<std::unique_ptr<IData>> cameras_;
    std::vector<std::unique_ptr<IAlgorithm>> algorithms_;
    std::unique_ptr<ISoC> soc_;
    std::unique_ptr<IDisplay> display_; // Add a display interface
*/
    //Share poits to the components
    std::vector<std::shared_ptr<IData>> cameras_;
    std::vector<std::shared_ptr<IAlgorithm>> algorithms_;
    std::shared_ptr<ISoC> soc_;
    std::shared_ptr<IDisplay> display_; // Add a display interface


    // Threads
    std::vector<std::thread> cameraThreads_;
    std::vector<std::thread> algorithmThreads_;
    std::thread socThread_;
    std::thread displayThread_; // Add a display Thread interface
    

    // Shared queue for frames
    SharedQueue<FrameData> frameQueue_;

    // Error callback
    std::function<void(const std::string&)> errorCallback_;

    int displayWidth_            = 320;//640;
    int displayHeight_           = 240;//480;

    // Buffer for processed data
    std::vector<uint8_t> processedRGB_;
    std::mutex           processedMutex_;

    std::shared_ptr<SoCConcrete> soc_;
    std::shared_ptr<AlgorithmConcrete> algo_;
    std::shared_ptr<DataConcrete> camera0_;
    std::shared_ptr<DataConcrete> camera1_;
};
//_______________________________________________________
// Implement the missing methods
//_______________________________________________________

SystemCaptureFactory::~SystemCaptureFactory() {
    stopCapture();
    if (displayThread_.joinable()) {
        displayThread_.join();
    }
}

void SystemCaptureFactory::initializeCapture() {
    try {
        // Create cameras, algorithms, SoC, display
        auto cam1   = modelingFactory_.createDataComponent();
        //auto cam2   = modelingFactory_.createDataComponent();
        auto alg1   = modelingFactory_.createAlgorithmComponent();
        auto alg2   = modelingFactory_.createAlgorithmComponent();
        auto socPtr = modelingFactory_.createSoCComponent();
        auto display = modelingFactory_.createDisplayComponent();

        // Set callbacks
        if (errorCallback_) {
            cam1->setErrorCallback(errorCallback_);
          //  cam2->setErrorCallback(errorCallback_);
            alg1->setErrorCallback(errorCallback_);
            alg2->setErrorCallback(errorCallback_);
            socPtr->setErrorCallback(errorCallback_);
            display->setErrorCallback(errorCallback_);
        }

        bool cam1Opened = cam1->openDevice("/dev/video0");
        if (!cam1Opened && errorCallback_) {
            errorCallback_("[SystemCaptureFactory] Failed to open /dev/video0");
        }
/*
        bool cam2Opened = cam2->openDevice("/dev/video1");
        if (!cam2Opened && errorCallback_) {
            errorCallback_("[SystemCaptureFactory] Failed to open /dev/video1");
        }
*/
        // Configure cameras
        CameraConfig defaultConfig;
        defaultConfig.width       = 640;
        defaultConfig.height      = 480;
        defaultConfig.fps         = 30;
        defaultConfig.pixelFormat = "YUYV";

        if (cam1Opened) cam1->configure(defaultConfig);
     //   if (cam2Opened) cam2->configure(defaultConfig);

        cameras_.push_back(std::move(cam1));
       // cameras_.push_back(std::move(cam2));
        algorithms_.push_back(std::move(alg1));
        algorithms_.push_back(std::move(alg2));
        soc_     = std::move(socPtr);
        display_ = std::move(display);

        soc_->initializeSoC();
        spdlog::info("[SystemCaptureFactory] SoC initialized.");

        // Start streaming
        for (auto& cam : cameras_) {
            spdlog::info("[SystemCaptureFactory] Starting streaming for camera {}...",cam);
            if (!cam->startStreaming()) {
                if (errorCallback_) {
                    errorCallback_("[SystemCaptureFactory] Failed to start streaming camera.");
                }
            }else{
                spdlog::info("[SystemCaptureFactory] Camera {} started streaming.",cam);
            }
        }

        // Start algorithms
        for (auto& alg : algorithms_) {
            alg->startAlgorithm();
        }

        // Initialize display
        spdlog::info("[SystemCaptureFactory] Initializing SDL display{} {})...", defaultConfig.width, defaultConfig.height);
        display_->initializeDisplay(defaultConfig.width, defaultConfig.height);

        // SoC monitor thread
        socThread_ = std::thread(&SystemCaptureFactory::monitorLoop, this, soc_.get());

        // Capture threads
        for (auto& cam : cameras_) {
            if (cam->isStreaming()) {
                spdlog::info("[SystemCaptureFactory] Spawning capture thread for camera {}...", cam);
                cameraThreads_.emplace_back(&SystemCaptureFactory::captureLoop, this, cam.get());
            } else if (errorCallback_) {
                errorCallback_("[SystemCaptureFactory] Not spawning capture thread; streaming is off.");
            }
        }

        // Algorithm threads
        for (auto& alg : algorithms_) {
            algorithmThreads_.emplace_back(&SystemCaptureFactory::algorithmLoop, this, alg.get());
        }

        // Display thread
        displayThread_ = std::thread(&SystemCaptureFactory::displayLoop, this);

        running_ = true;
        spdlog::info("[SystemCaptureFactory] System fully initialized.");

    } catch (const std::exception& e) {
        std::cerr << "[SystemCaptureFactory] Initialization error: " << e.what() << std::endl;
        stopCapture(); // cleanup
        throw; 
    }
}

void SystemCaptureFactory::stopCapture() {
    if (!running_) return;
    spdlog::info("[SystemCaptureFactory] Stopping system...");
    running_ = false;

    // Join camera threads
    for (auto& t : cameraThreads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    cameraThreads_.clear();

    // Stop streaming
    for (auto& cam : cameras_) {
        cam->stopStreaming();
    }

    // SoC
    if (soc_) {
        soc_->stopSoC();
        if (socThread_.joinable()) {
            socThread_.join();
        }
    }

    // Stop algorithms
    for (auto& alg : algorithms_) {
        alg->stopAlgorithm();
    }
    for (auto& t : algorithmThreads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    algorithmThreads_.clear();

    // Stop / close display
    if (display_) {
        display_->closeDisplay();
    }
    if (displayThread_.joinable()) {
        displayThread_.join();
    }

    frameQueue_.stop(); // If you have a method to stop the shared queue

    spdlog::info("[SystemCaptureFactory] System stopped.");
}

void SystemCaptureFactory::pauseAll() {
    paused_ = true;
    spdlog::warn("[SystemCaptureFactory] Paused all threads.");
}

void SystemCaptureFactory::resumeAll() {
    paused_ = false;
    spdlog::warn("[SystemCaptureFactory] Resumed all threads.");
}

void SystemCaptureFactory::setThreadAffinity(int /*cpuCore*/) {
    // not implemented
}

void SystemCaptureFactory::configureCamera(size_t /*index*/, const CameraConfig& /*cfg*/) {
    // not implemented
}

void SystemCaptureFactory::configureAlgorithm(size_t /*index*/, const AlgorithmConfig& /*cfg*/) {
    // not implemented
}

void SystemCaptureFactory::configureSoC(const SoCConfig& /*cfg*/) {
    // not implemented
}

void SystemCaptureFactory::setGlobalErrorCallback(std::function<void(const std::string&)> callback) {
    errorCallback_ = std::move(callback);
}

//----------------------------------------
// captureLoop
//----------------------------------------
void SystemCaptureFactory::captureLoop(IData* camera) {
    // Adjust these if you want to dynamically set from camera config
    const int capWidth  = 320;//640;
    const int capHeight = 240;//480;
    std::vector<uint8_t> rgbOriginal(capWidth * capHeight * 3, 0);

    while (running_) {
        if (paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        void*  dataPtr   = nullptr;
        size_t sizeBytes = 0;

        if (!camera->dequeFrame(dataPtr, sizeBytes)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        if (!dataPtr) {
            spdlog::error("[captureLoop] Null frame data received. Skipping frame.");
            continue;
        }

        // Convert YUYV → RGB
        yuyvToRGB((uint8_t*)dataPtr, rgbOriginal.data(), capWidth, capHeight);

        // Show the original frame on display
        display_->updateOriginalFrame(rgbOriginal.data(), capWidth, capHeight);

        // Push to the frame queue for the algorithm
        FrameData frame;
        frame.data      = dataPtr;
        frame.size      = sizeBytes;
        frame.timestamp = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()
                          ).count();

        frameQueue_.push(frame);

        camera->queueBuffer();
    }
    spdlog::info("[SystemCaptureFactory] captureLoop ended.");
}

//----------------------------------------
// monitorLoop
//----------------------------------------
void SystemCaptureFactory::monitorLoop(ISoC* soc) {
    while (running_) {
        if (paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        std::string perf = soc->getPerformance();
        if (!perf.empty()) {
            spdlog::info("[SoCConcrete Stats] {}", perf);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    spdlog::info("[SystemCaptureFactory] monitorLoop ended.");
}

//----------------------------------------
// displayLoop
//----------------------------------------
void SystemCaptureFactory::displayLoop() {
    while (running_) {
        if (paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        if (display_) {
            display_->renderAndPollEvents();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    spdlog::info("[SystemCaptureFactory] displayLoop ended.");
}

//----------------------------------------
// algorithmLoop
//----------------------------------------
void SystemCaptureFactory::algorithmLoop(IAlgorithm* alg) {
    // local buffer for “processed” frame
    const int capWidth  = 320;//640;
    const int capHeight = 240;//480;
    std::vector<uint8_t> localProcessed(capWidth * capHeight * 3, 0);

    while (running_) {
        if (paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        FrameData frame;
        if (frameQueue_.pop(frame)) {
            spdlog::info("[algorithmLoop] Processing frame, size={} bytes.", frame.size);

            bool ok = alg->processFrame(frame);
            if (!ok && errorCallback_) {
                errorCallback_("[SystemCaptureFactory] algorithmLoop: frame processing failed.");
            } else {
                // fill localProcessed or do real processing
                for (size_t i = 0; i < localProcessed.size(); i += 3) {
                    localProcessed[i + 0] = 0;   // R
                    localProcessed[i + 1] = 128; // G
                    localProcessed[i + 2] = 255; // B
                }

                // Update “processed” texture on display
                {
                    std::lock_guard<std::mutex> lock(processedMutex_);
                    processedRGB_.assign(localProcessed.begin(), localProcessed.end());
                }
                display_->updateProcessedFrame(localProcessed.data(), capWidth, capHeight);
                spdlog::info("[algorithmLoop] Frame processed successfully.");
            }
        } else {
            spdlog::debug("[algorithmLoop] Frame queue empty, sleeping...");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    spdlog::info("[SystemCaptureFactory] algorithmLoop ended.");
}

// Compare this snippet from Stage_01/Includes/SystemCaptureFactory.h: