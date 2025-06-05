
// SampleTestIntegrationCode_v3.cpp
// -----------------------------------------------------------------------------
// End-to-end pipeline test wired to SystemMetricsAggregatorConcreteV2.
// * One CSV/JSON line per frame, with derived latency + energy metrics.
// * No manual pushMetrics in the main loop ? modules handle Begin/Merge/Flush.
// * Supports nested JSON config, profiling, system behavior, and enhanced validation.
// -----------------------------------------------------------------------------

#include "Stage_01/Concretes/SystemMetricsAggregatorConcrete_V2.h"
#include "Stage_01/Concretes/DataConcrete_new.h"
#include "Stage_01/Concretes/AlgorithmConcrete_new.h"
#include "Stage_01/Concretes/SdlDisplayConcrete_new.h"
#include "Stage_01/Concretes/SoCConcrete_new.h"
#include "Stage_01/Concretes/LynsynMonitorConcrete.h"

#include "Stage_01/SharedStructures/CameraConfig.h"
#include "Stage_01/SharedStructures/DisplayConfig.h"
#include "Stage_01/SharedStructures/AlgorithmConfig.h"
#include "Stage_01/SharedStructures/SoCConfig.h"
#include "Stage_01/SharedStructures/LynsynMonitorConfig.h"
#include "Stage_01/SharedStructures/ThreadManager.h"
#include "Stage_01/SharedStructures/SharedQueue.h"
#include "Stage_01/SharedStructures/ZeroCopyFrameData.h"

#include "../../usr/local/cuda-10.2/include/cuda_runtime.h"
#include "../../usr/local/cuda-10.2/include/device_launch_parameters.h"

#include "../nlohmann/json.hpp"
#include <spdlog/spdlog.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <stdexcept>
#include <SDL2/SDL.h>
#include <unistd.h>
#include <map>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
using json = nlohmann::json;

// Pixel format string to enum conversion
PixelFormat stringToPixelFormat(const std::string& format) {
    if (format == "YUYV") return PixelFormat::YUYV;
    if (format == "MJPG") return PixelFormat::MJPG;
    throw std::invalid_argument("Unsupported pixel format: " + format);
}

// Algorithm type string to enum conversion
AlgorithmType stringToAlgorithmType(const std::string& type) {
    static const std::map<std::string, AlgorithmType> typeMap = {
        {"MultiThreadedInvert", AlgorithmType::MultiThreadedInvert},
        {"GaussianBlur", AlgorithmType::GaussianBlur},
        {"OpticalFlow_LucasKanade", AlgorithmType::OpticalFlow_LucasKanade},
        {"MatrixMultiply", AlgorithmType::MatrixMultiply},
        {"Mandelbrot", AlgorithmType::Mandelbrot},
        {"GPUMatrixMultiply", AlgorithmType::GPUMatrixMultiply},
        {"SobelEdge", AlgorithmType::SobelEdge},
        {"MedianFilter", AlgorithmType::MedianFilter},
        {"HistogramEqualization", AlgorithmType::HistogramEqualization},
        {"HeterogeneousGaussianBlur", AlgorithmType::HeterogeneousGaussianBlur}
    };
    auto it = typeMap.find(type);
    if (it == typeMap.end()) {
        throw std::invalid_argument("Unsupported algorithm type: " + type);
    }
    return it->second;
}

// Validation helper functions
namespace Validation {
    bool validateSoCConfig(const SoCConfig& cfg) {
        if (!cfg.validate()) {
            spdlog::error("Invalid SoC configuration");
            return false;
        }
        // std::string checkCmd = cfg.customCommand + " --version 2>&1";
        // if (std::system(checkCmd.c_str()) != 0) {
        //     spdlog::error("tegrastats command not found: {}", cfg.customCommand);
        //     return false;
        // }
        return true;
    }

    bool validateCameraConfig(const CameraConfig& cfg) {
        if (!cfg.validate()) {
            spdlog::error("Invalid camera configuration");
            return false;
        }
        if (cfg.pixelFormat != PixelFormat::YUYV && cfg.pixelFormat != PixelFormat::MJPG) {
            spdlog::error("Unsupported pixel format: {}", static_cast<int>(cfg.pixelFormat));
            return false;
        }
        if (access("/dev/video0", F_OK) != 0) {
            spdlog::error("Camera device /dev/video0 not found");
            return false;
        }
        return true;
    }

    bool validateDisplayConfig(const DisplayConfig& cfg) {
        if (!cfg.validate()) {
            spdlog::error("Invalid display configuration");
            return false;
        }
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            spdlog::error("SDL initialization failed: {}", SDL_GetError());
            return false;
        }
        SDL_Quit();
        return true;
    }

    bool validateAlgorithmConfig(const AlgorithmConfig& cfg) {
        if (!cfg.validate()) {
            spdlog::error("Invalid algorithm configuration");
            return false;
        }
        if (cfg.useGPU) {
            // Placeholder CUDA check
            // if (!cudaDeviceAvailable()) {
            //     cudaDeviceGetAttribute(&cudaDeviceProp prop, cudaDevAttrComputeCapabilityMajor, 0);
            //     spdlog::error("GPU required but not available");
            //     return false;
            // }
        }
        return true;
    }

    bool validateLynsynConfig(const LynsynMonitorConfig& cfg) {
        if (!cfg.validate()) {
            spdlog::error("Invalid Lynsyn configuration");
            return false;
        }
        fs::path path(cfg.outputCSV);
        auto dir = path.parent_path();
        if (!dir.empty() && !fs::exists(dir)) {
            spdlog::error("Output directory does not exist: {}", dir.string());
            return false;
        }
        return true;
    }

    bool validateQueueCapacity(size_t capacity, int fps) {
        size_t minCapacity = fps * 2;
        if (capacity < minCapacity) {
            spdlog::error("Queue capacity {} too low, minimum required: {}", capacity, minCapacity);
            return false;
        }
        return true;
    }

    bool validateProfilingConfig(const nlohmann::json& cfg) {
        if (!cfg.contains("metricsOutputFile") || !cfg["metricsOutputFile"].is_string()) {
            spdlog::error("Invalid or missing profiling metricsOutputFile");
            return false;
        }
        if (!cfg.contains("samplingIntervalMs") || !cfg["samplingIntervalMs"].is_number_integer() || cfg["samplingIntervalMs"] <= 0) {
            spdlog::error("Invalid or missing profiling samplingIntervalMs");
            return false;
        }
        return true;
    }

    bool validateSystemBehavior(const nlohmann::json& cfg) {
        if (!cfg.contains("autoRunDurationSec") || !cfg["autoRunDurationSec"].is_number() || cfg["autoRunDurationSec"] <= 0) {
            spdlog::error("Invalid or missing autoRunDurationSec");
            return false;
        }
        if (!cfg.contains("pauseDurationSec") || !cfg["pauseDurationSec"].is_number() || cfg["pauseDurationSec"] <= 0) {
            spdlog::error("Invalid or missing pauseDurationSec");
            return false;
        }
        return true;
    }
}

// The Main function!!
//======================================================================================
int main(int argc, char* argv[]) {
    // Line 193: Fix spdlog formatting with current date and time
    auto now = std::chrono::system_clock::now();
    std::time_t now_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_t), "%Y-%m-%d %H:%M:%S");
    SPDLOG_INFO("Starting ZeroCopy Pipeline Test at {}", ss.str());

    // Command-line runtime duration (override JSON if provided)
    double runSeconds = 60.0;
    if (argc > 1) {
        try {
            runSeconds = std::stod(argv[1]);
            if (runSeconds <= 0) throw std::invalid_argument("Runtime must be positive");
        } catch (const std::exception& e) {
            spdlog::error("Invalid runtime argument: {}. Using default (60s)", e.what());
        }
    }

    // Load configuration from JSON
    nlohmann::json config;
    try {
        if (fs::exists("config.json")) {
            std::ifstream configFile("config.json");
            config = nlohmann::json::parse(configFile);
            spdlog::info("Loaded configuration from config.json");
        } else {
            spdlog::warn("config.json not found. Using defaults");
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse config.json: {}. Using defaults", e.what());
    }

    // Validate profiling and system behavior
    if (config.contains("Profiling") && !Validation::validateProfilingConfig(config["Profiling"])) {
        throw std::runtime_error("Invalid profiling configuration");
    }
    if (config.contains("SystemBehavior") && !Validation::validateSystemBehavior(config["SystemBehavior"])) {
        throw std::runtime_error("Invalid system behavior configuration");
    }

    // Override runSeconds from JSON if not provided via command line
    if (argc <= 1 && config.contains("SystemBehavior") && config["SystemBehavior"].contains("autoRunDurationSec")) {
        runSeconds = config["SystemBehavior"]["autoRunDurationSec"];
    }

    // RAII cleanup struct
    struct PipelineCleanup {
        ThreadManager* tm = nullptr;
        DataConcrete* camera = nullptr;
        AlgorithmConcrete* algorithm = nullptr;
        SdlDisplayConcrete* display = nullptr;
        LynsynMonitorConcrete* powerMon = nullptr;
        SoCConcrete* soc = nullptr;

        ~PipelineCleanup() {
            spdlog::info("Initiating pipeline shutdown...");
            if (display) { display->closeDisplay(); spdlog::info("Display stopped"); }
            if (algorithm) { algorithm->stopAlgorithm(); spdlog::info("Algorithm stopped"); }
            if (camera) {
                camera->stopCapture();
                camera->stopStreaming();
                camera->closeDevice();
                spdlog::info("Camera stopped");
            }
            if (powerMon) { powerMon->stop(); spdlog::info("Power monitor stopped"); }
            if (soc) { soc->stopSoC(); spdlog::info("SoC monitor stopped"); }
            if (tm) { tm->shutdown(); spdlog::info("Thread manager stopped"); }
        }
    } cleanup;

    try {
        //------------------------------------------------------------------
        // 0. Metrics aggregator
        //------------------------------------------------------------------
        auto agg = std::make_shared<SystemMetricsAggregatorConcreteV2>();
        if (!agg) throw std::runtime_error("Failed to create metrics aggregator");
        agg->setCsvPath(config.value("metrics_csv", "realtime_metrics_01.csv"));
        agg->setJsonPath(config.value("metrics_json", "realtime_metrics.ndjson"));
        agg->setRetentionWindow(std::chrono::seconds(config.value("retention_window_sec", 120)));
        spdlog::info("Metrics aggregator initialized");

        //------------------------------------------------------------------
        // 1. Queues & thread manager
        //------------------------------------------------------------------
        cleanup.tm = new ThreadManager();
        auto cam2Alg = std::make_shared<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>(60);
        auto cam2Disp = std::make_shared<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>(60);
        auto alg2Disp = std::make_shared<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>(50);

        int cameraFps = config.contains("CameraConfig") ? config["CameraConfig"].value("fps", 30) : 30;
        if (!Validation::validateQueueCapacity(60, cameraFps)) {
            throw std::runtime_error("Invalid queue capacity");
        }
        spdlog::info("Queues initialized with capacity 15");
        spdlog::info("Thread pool started with {} threads", cleanup.tm->getThreadCount());

        //------------------------------------------------------------------
        // 2. SoC + Power monitors
        //------------------------------------------------------------------
        cleanup.soc = new SoCConcrete(agg);
        SoCConfig socCfg;
        if (config.contains("SoCConfig")) {
            socCfg.pollIntervalMs = config["SoCConfig"].value("pollIntervalMs", 1000);
            socCfg.customCommand = config["SoCConfig"].value("customCommand", "tegrastats --interval 1000");
            socCfg.exportCSV = config["SoCConfig"].value("exportCSV", false);
            socCfg.csvPath = config["SoCConfig"].value("csvPath", "jetson_nano_tegrastats.csv");
        }
        if (!Validation::validateSoCConfig(socCfg)) {
            throw std::runtime_error("Invalid SoC configuration");
        }
        if (!cleanup.soc->configure(socCfg)) {
            throw std::runtime_error("SoC configuration failed");
        }
        int retries = 3;
        bool socInitialized = false;
        while (retries-- > 0 && !socInitialized) {
            if (cleanup.soc->initializeSoC()) {
                socInitialized = true;
                spdlog::info("SoC monitor initialized");
            } else {
                spdlog::warn("SoC initialization failed, retrying ({}/3)...", 3 - retries);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        if (!socInitialized) {
            throw std::runtime_error("SoC initialization failed after retries");
        }

        cleanup.powerMon = new LynsynMonitorConcrete(agg, *cleanup.tm);
        LynsynMonitorConfig pCfg;
        if (config.contains("LynsynConfig")) {
            pCfg.periodSampling = config["LynsynConfig"].value("periodSampling", true);
            pCfg.outputCSV = config["LynsynConfig"].value("outputCSV", "power_raw.csv");
            pCfg.durationSec = config["LynsynConfig"].value("durationSec", 1.0);
            pCfg.coreMask = config["LynsynConfig"].value("coreMask", 0xF);
        }
        if (!Validation::validateLynsynConfig(pCfg)) {
            throw std::runtime_error("Invalid Lynsyn configuration");
        }
        if (!cleanup.powerMon->configure(pCfg)) {
            throw std::runtime_error("Lynsyn configuration failed");
        }
        if (!cleanup.powerMon->initialize()) {
            throw std::runtime_error("Lynsyn monitor initialization failed");
        }
        cleanup.powerMon->startMonitoring();
        spdlog::info("Power monitor started");

        //------------------------------------------------------------------
        // 3. Camera
        //------------------------------------------------------------------
        cleanup.camera = new DataConcrete(cam2Alg, cam2Disp, agg);
        CameraConfig camCfg;
        if (config.contains("CameraConfig")) {
            camCfg.width = config["CameraConfig"].value("width", 640);
            camCfg.height = config["CameraConfig"].value("height", 480);
            camCfg.fps = config["CameraConfig"].value("fps", 30);
            camCfg.pixelFormat = stringToPixelFormat(config["CameraConfig"].value("pixelFormat", "YUYV"));
        } else {
            camCfg = CameraConfig(640, 480, 30, PixelFormat::YUYV);
        }
        if (!Validation::validateCameraConfig(camCfg)) {
            throw std::runtime_error("Invalid camera configuration");
        }
        if (!cleanup.camera->openDevice("/dev/video0")) {
            throw std::runtime_error("Failed to open camera device");
        }
        if (!cleanup.camera->configure(camCfg)) {
            throw std::runtime_error("Camera configuration failed");
        }
        if (!cleanup.camera->startStreaming()) {
            throw std::runtime_error("Camera streaming failed");
        }
        cleanup.camera->startCapture();
        spdlog::info("Camera started ({}x{}, {} FPS)", camCfg.width, camCfg.height, camCfg.fps);

        //------------------------------------------------------------------
        // 4. Algorithm
        //------------------------------------------------------------------
        AlgorithmConfig algCfg;
        if (config.contains("AlgorithmConfig")) {
            algCfg.algorithmType = stringToAlgorithmType(config["AlgorithmConfig"].value("algorithmType", "MultiThreadedInvert"));
            algCfg.concurrencyLevel = config["AlgorithmConfig"].value("concurrencyLevel", 1);
            algCfg.useGPU = config["AlgorithmConfig"].value("useGPU", false);
            algCfg.blurRadius = config["AlgorithmConfig"].value("blurRadius", 5);
            if (config["AlgorithmConfig"].contains("opticalFlowConfig")) {
                algCfg.opticalFlowConfig.windowSize = config["AlgorithmConfig"]["opticalFlowConfig"].value("windowSize", 15);
                algCfg.opticalFlowConfig.maxLevel = config["AlgorithmConfig"]["opticalFlowConfig"].value("maxLevel", 2);
            }
        }
        if (!Validation::validateAlgorithmConfig(algCfg)) {
            throw std::runtime_error("Invalid algorithm configuration");
        }
        try {
            cleanup.algorithm = new AlgorithmConcrete(cam2Alg, alg2Disp, *cleanup.tm, agg);
            if (!cleanup.algorithm) {
                throw std::runtime_error("Algorithm creation failed");
            }
            cleanup.algorithm->startAlgorithm();
            spdlog::info("Algorithm started (type: {})", 
                        config["AlgorithmConfig"].value("algorithmType", "MultiThreadedInvert"));
        } catch (const std::exception& e) {
            delete cleanup.algorithm;
            cleanup.algorithm = nullptr;
            throw std::runtime_error(std::string("Algorithm initialization failed: ") + e.what());
        }

        //------------------------------------------------------------------
        // 5. Display
        //------------------------------------------------------------------
        cleanup.display = new SdlDisplayConcrete(cam2Disp, alg2Disp, agg);
        DisplayConfig dspCfg;
        if (config.contains("DisplayConfig")) {
            dspCfg = DisplayConfig(
                config["DisplayConfig"].value("width", 640),
                config["DisplayConfig"].value("height", 480),
                config["DisplayConfig"].value("fullScreen", false),
                config["DisplayConfig"].value("windowTitle", "ZeroCopy Pipeline")
            );
        } else {
            dspCfg = DisplayConfig(640, 480, false, "ZeroCopy Pipeline");
        }
        if (!Validation::validateDisplayConfig(dspCfg)) {
            throw std::runtime_error("Invalid display configuration");
        }
        if (!cleanup.display->configure(dspCfg)) {
            throw std::runtime_error("Display configuration failed");
        }
        if (!cleanup.display->initializeDisplay(dspCfg.width, dspCfg.height)) {
            throw std::runtime_error("Display initialization failed");
        }
        spdlog::info("Display initialized ({}x{})", dspCfg.width, dspCfg.height);

        //------------------------------------------------------------------
        // 6. Main loop
        //------------------------------------------------------------------
        // Added main loop to continuously render frames and handle pipeline flow
        // Addressed issue where program stopped after display initialization
        auto start = std::chrono::steady_clock::now();
        size_t frameCount = 0;
        auto lastLogTime = start;
        double samplingIntervalMs = config.contains("Profiling") ? config["Profiling"].value("samplingIntervalMs", 1000) : 1000;
        double pauseDurationSec = config.contains("SystemBehavior") ? config["SystemBehavior"].value("pauseDurationSec", 2.0) : 2.0;
        bool paused = false;
        spdlog::debug("Starting main loop with runSeconds={}", runSeconds);
        while (cleanup.display->is_Running()) {
            auto now = std::chrono::steady_clock::now();
            if (!paused) {
                // Modification (2025-06-01): Fix compilation error - Changed `pop()` to `try_pop()` to match SharedQueue interface
                // `pop()` requires an argument and doesn't return the item directly; `try_pop` is non-blocking and better for display loop
                std::shared_ptr<ZeroCopyFrameData> frame;
                if (alg2Disp->try_pop(frame)) {
                    // Modification (2025-06-01): Fix compilation error - Changed `frame->data` to `frame->dataPtr` to match ZeroCopyFrameData member
                    cleanup.display->renderFrame_1(frame->dataPtr,frame->width,frame->height); // Update display buffer with processed frame
                    frameCount++;
                    // Modification (2025-06-01): Fix logging - Changed `frame->data.size()` to `frame->size` as `dataPtr` is void* and size is a member
                    spdlog::debug("[Main] Rendered frame: width={}, height={}, size={}", 
                         frame->width, frame->height, frame->size);
                } else {
                    //spdlog::debug("No frame available in alg2Disp queue");
                    spdlog::debug("[Main] No frame available in alg2Disp queue, checking display events");
                }
                cleanup.display->renderAndPollEvents(); // Handle SDL events and render
            }

            // Health check
            if (!cleanup.soc->isMonitoringActive()) {
                spdlog::error("SoC monitor stopped unexpectedly");
                break;
            }
            if (!cleanup.camera->isStreaming()) {
                spdlog::error("Camera streaming stopped unexpectedly");
                break;
            }

            // Performance logging
            if (std::chrono::duration<double>(now - lastLogTime).count() * 1000 >= samplingIntervalMs) {
                double fps = frameCount / std::chrono::duration<double>(now - start).count();
                spdlog::info("Performance: FPS={:.2f}, Queues: cam2Alg={}, cam2Disp={}, alg2Disp={}",
                            fps, cam2Alg->size(), cam2Disp->size(), alg2Disp->size());
                // Export profiling metrics
                if (config.contains("Profiling")) {
                    std::ofstream profFile(config["Profiling"].value("metricsOutputFile", "PerformanceMetrics.csv"), std::ios::app);
                    auto now_time = std::chrono::system_clock::now();
                    auto now_c = std::chrono::system_clock::to_time_t(now_time);
                    std::stringstream ss;
                    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
                    profFile << ss.str() << "," << fps << ","
                            << cam2Alg->size() << "," << cam2Disp->size() << "," << alg2Disp->size() << "\n";
                }
                lastLogTime = now;
            }

            // Queue overflow mitigation
            if (cam2Alg->isFull() || cam2Disp->isFull() || alg2Disp->isFull()) {
                spdlog::warn("Queue overflow: cam2Alg={}, cam2Disp={}, alg2Disp={}",
                            cam2Alg->isFull(), cam2Disp->isFull(), alg2Disp->isFull());
                cleanup.camera->pauseCapture();
                paused = true;
                std::this_thread::sleep_for(std::chrono::duration<double>(pauseDurationSec));
                cleanup.camera->resumeCapture();
                paused = false;
            }

            auto elapsed = std::chrono::duration<double>(now - start).count();
            if (elapsed > runSeconds) {
                spdlog::info("Runtime limit ({}s) reached", runSeconds);
                break;
            }
            // Modification (2025-06-02 06:57 AM -04): Removed 10ms sleep to avoid frame rate conflict with renderAndPollEvents
            //std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Control frame rate
        }

        //------------------------------------------------------------------
        // 7. Shutdown
        //------------------------------------------------------------------
        agg->exportToCSV("final_metrics.csv");
        agg->exportToJSON("final_metrics.json");
        spdlog::info("Metrics exported to final_metrics.csv and final_metrics.json");

    } catch (const std::exception& e) {
        spdlog::error("Fatal error in pipeline: {}", e.what());
        return 1;
    }

    spdlog::info("Pipeline test completed successfully");
    return 0;
    

}

// SampleTestIntegrationCode_v3.cpp

// // -----------------------------------------------------------------------------
// // End-to-end pipeline test wired to SystemMetricsAggregatorConcreteV2.
// // * One CSV/JSON line per frame, with derived latency + energy metrics.
// // * No manual pushMetrics in the main loop ? modules handle Begin/Merge/Flush.
// // * Supports nested JSON config, }}profiling, system behavior, and enhanced validation.
// // -----------------------------------------------------------------------------

// #include "Stage_01/Concretes/SystemMetricsAggregatorConcrete_V2.h"
// #include "Stage_01/Concretes/DataConcrete_new.h"
// #include "Stage_01/Concretes/AlgorithmConcrete_new.h"
// #include "Stage_01/Concretes/SdlDisplayConcrete_new.h"
// #include "Stage_01/Concretes/SoCConcrete_new.h"
// #include "Stage_01/Concretes/LynsynMonitorConcrete.h"

// #include "Stage_01/SharedStructures/CameraConfig.h"
// #include "Stage_01/SharedStructures/DisplayConfig.h"
// #include "Stage_01/SharedStructures/AlgorithmConfig.h"
// #include "Stage_01/SharedStructures/SoCConfig.h"
// #include "Stage_01/SharedStructures/LynsynMonitorConfig.h"
// #include "Stage_01/SharedStructures/ThreadManager.h"
// #include "Stage_01/SharedStructures/SharedQueue.h"
// #include "Stage_01/SharedStructures/ZeroCopyFrameData.h"

// #include "../../usr/local/cuda-10.2/include/cuda_runtime.h"
// #include "../../usr/local/cuda-10.2/include/device_launch_parameters.h"

// #include "../nlohmann/json.hpp"
// //#inclide "../nlohmann/json.hpp"
// #include <spdlog/spdlog.h>
// //#include <nlohmann/json.hpp>
// #include <iostream>

// #include <sstream>  // Add at the top of the file
// #include <iomanip>  // Add at the top of the file

// #include <fstream>
// #include <thread>
// #include <chrono>
// #include <atomic>
// #include <filesystem>
// #include <fstream>
// #include <stdexcept>
// #include <SDL2/SDL.h>
// #include <unistd.h>
// #include <map>
// #include <bits/fs_fwd.h>


// #include <experimental/filesystem>  // Make sure this is included
// #include <c++/10/bits/fs_fwd.h>
// namespace fs = std::experimental::filesystem;
// using json = nlohmann::json;

// // Pixel format string to enum conversion
// PixelFormat stringToPixelFormat(const std::string& format) {
//     if (format == "YUYV") return PixelFormat::YUYV;
//     if (format == "MJPG") return PixelFormat::MJPG;
//     throw std::invalid_argument("Unsupported pixel format: " + format);
// }

// // Algorithm type string to enum conversion
// AlgorithmType stringToAlgorithmType(const std::string& type) {
//     static const std::map<std::string, AlgorithmType> typeMap = {
//         {"MultiThreadedInvert", AlgorithmType::MultiThreadedInvert},
//         {"GaussianBlur", AlgorithmType::GaussianBlur},
//         {"OpticalFlow_LucasKanade", AlgorithmType::OpticalFlow_LucasKanade},
//         {"MatrixMultiply", AlgorithmType::MatrixMultiply},
//         {"Mandelbrot", AlgorithmType::Mandelbrot},
//         {"GPUMatrixMultiply", AlgorithmType::GPUMatrixMultiply},
//         {"SobelEdge", AlgorithmType::SobelEdge},
//         {"MedianFilter", AlgorithmType::MedianFilter},
//         {"HistogramEqualization", AlgorithmType::HistogramEqualization},
//         {"HeterogeneousGaussianBlur", AlgorithmType::HeterogeneousGaussianBlur}
//     };
//     auto it = typeMap.find(type);
//     if (it == typeMap.end()) {
//         throw std::invalid_argument("Unsupported algorithm type: " + type);
//     }
//     return it->second;
// }

// // Validation helper functions
// namespace Validation {
//     bool validateSoCConfig(const SoCConfig& cfg) {
//         if (!cfg.validate()) {
//             spdlog::error("Invalid SoC configuration");
//             return false;
//         }
//         std::string checkCmd = cfg.customCommand + " --version 2>&1";
//         if (std::system(checkCmd.c_str()) != 0) {
//             spdlog::error("tegrastats command not found: {}", cfg.customCommand);
//             return false;
//         }
//         return true;
//     }

//     bool validateCameraConfig(const CameraConfig& cfg) {
//         if (!cfg.validate()) {
//             spdlog::error("Invalid camera configuration");
//             return false;
//         }
//         if (cfg.pixelFormat != PixelFormat::YUYV && cfg.pixelFormat != PixelFormat::MJPG) {
//             spdlog::error("Unsupported pixel format: {}", static_cast<int>(cfg.pixelFormat));
//             return false;
//         }
//         if (access("/dev/video0", F_OK) != 0) {
//             spdlog::error("Camera device /dev/video0 not found");
//             return false;
//         }
//         return true;
//     }

//     bool validateDisplayConfig(const DisplayConfig& cfg) {
//         if (!cfg.validate()) {
//             spdlog::error("Invalid display configuration");
//             return false;
//         }
//         if (SDL_Init(SDL_INIT_VIDEO) < 0) {
//             spdlog::error("SDL initialization failed: {}", SDL_GetError());
//             return false;
//         }
//         SDL_Quit();
//         return true;
//     }

//     bool validateAlgorithmConfig(const AlgorithmConfig& cfg) {
//         if (!cfg.validate()) {
//             spdlog::error("Invalid algorithm configuration");
//             return false;
//         }
//         if (cfg.useGPU) {
//             // Placeholder CUDA check
//             // if (!cudaDeviceAvailable()) {
//             //     cudaDeviceGetAttribute(&cudaDeviceProp prop, cudaDevAttrComputeCapabilityMajor, 0);
//             //     spdlog::error("GPU required but not available");
//             //     return false;
//             // }
//         }
//         return true;
//     }

//     bool validateLynsynConfig(const LynsynMonitorConfig& cfg) {
//         if (!cfg.validate()) {
//             spdlog::error("Invalid Lynsyn configuration");
//             return false;
//         }
//         fs::path path(cfg.outputCSV);
//         auto dir = path.parent_path();
//         if (!dir.empty() && !fs::exists(dir)) {
//             spdlog::error("Output directory does not exist: {}", dir.string());
//             return false;
//         }
//         return true;
//     }

//     bool validateQueueCapacity(size_t capacity, int fps) {
//         size_t minCapacity = fps * 2;
//         if (capacity < minCapacity) {
//             spdlog::error("Queue capacity {} too low, minimum required: {}", capacity, minCapacity);
//             return false;
//         }
//         return true;
//     }

//     bool validateProfilingConfig(const nlohmann::json& cfg) {
//         if (!cfg.contains("metricsOutputFile") || !cfg["metricsOutputFile"].is_string()) {
//             spdlog::error("Invalid or missing profiling metricsOutputFile");
//             return false;
//         }
//         if (!cfg.contains("samplingIntervalMs") || !cfg["samplingIntervalMs"].is_number_integer() || cfg["samplingIntervalMs"] <= 0) {
//             spdlog::error("Invalid or missing profiling samplingIntervalMs");
//             return false;
//         }
//         return true;
//     }

//     bool validateSystemBehavior(const nlohmann::json& cfg) {
//         if (!cfg.contains("autoRunDurationSec") || !cfg["autoRunDurationSec"].is_number() || cfg["autoRunDurationSec"] <= 0) {
//             spdlog::error("Invalid or missing autoRunDurationSec");
//             return false;
//         }
//         if (!cfg.contains("pauseDurationSec") || !cfg["pauseDurationSec"].is_number() || cfg["pauseDurationSec"] <= 0) {
//             spdlog::error("Invalid or missing pauseDurationSec");
//             return false;
//         }
//         return true;
//     }
// }
// // The Main function!!
// //======================================================================================
// int main(int argc, char* argv[]) {
//     // spdlog::set_level(spdlog::level::info);
//     // spdlog::info("Starting ZeroCopy Pipeline Test at {}", std::chrono::system_clock::now());

//     // Line 193: Fix spdlog formatting
//     auto now = std::chrono::system_clock::now();
//     std::time_t now_t = std::chrono::system_clock::to_time_t(now);
//     std::stringstream ss;
//     ss << std::put_time(std::localtime(&now_t), "%Y-%m-%d %H:%M:%S");
//     SPDLOG_INFO("Starting ZeroCopy Pipeline Test at {}", ss.str());

//     // Command-line runtime duration (override JSON if provided)
//     double runSeconds = 60.0;
//     if (argc > 1) {
//         try {
//             runSeconds = std::stod(argv[1]);
//             if (runSeconds <= 0) throw std::invalid_argument("Runtime must be positive");
//         } catch (const std::exception& e) {
//             spdlog::error("Invalid runtime argument: {}. Using default (60s)", e.what());
//         }
//     }

//     // Load configuration from JSON
//     nlohmann::json config;
//     try {
//         if (
//            fs::exists("config.json")) {
//             std::ifstream configFile("config.json");
//             config = nlohmann::json::parse(configFile);
//             spdlog::info("Loaded configuration from config.json");
//         } else {
//             spdlog::warn("config.json not found. Using defaults");
//         }
//     } catch (const std::exception& e) {
//         spdlog::error("Failed to parse config.json: {}. Using defaults", e.what());
//     }

//     // Validate profiling and system behavior
//     if (config.contains("Profiling") && !Validation::validateProfilingConfig(config["Profiling"])) {
//         throw std::runtime_error("Invalid profiling configuration");
//     }
//     if (config.contains("SystemBehavior") && !Validation::validateSystemBehavior(config["SystemBehavior"])) {
//         throw std::runtime_error("Invalid system behavior configuration");
//     }

//     // Override runSeconds from JSON if not provided via command line
//     if (argc <= 1 && config.contains("SystemBehavior") && config["SystemBehavior"].contains("autoRunDurationSec")) {
//         runSeconds = config["SystemBehavior"]["autoRunDurationSec"];
//     }

//     // RAII cleanup struct
//     struct PipelineCleanup {
//         ThreadManager* tm = nullptr;
//         DataConcrete* camera = nullptr;
//         AlgorithmConcrete* algorithm = nullptr;
//         SdlDisplayConcrete* display = nullptr;
//         LynsynMonitorConcrete* powerMon = nullptr;
//         SoCConcrete* soc = nullptr;

//         ~PipelineCleanup() {
//             spdlog::info("Initiating pipeline shutdown...");
//             if (display) { display->closeDisplay(); spdlog::info("Display stopped"); }
//             if (algorithm) { algorithm->stopAlgorithm(); spdlog::info("Algorithm stopped"); }
//             if (camera) {
//                 camera->stopCapture();
//                 camera->stopStreaming();
//                 camera->closeDevice();
//                 spdlog::info("Camera stopped");
//             }
//             if (powerMon) { powerMon->stop(); spdlog::info("Power monitor stopped"); }
//             if (soc) { soc->stopSoC(); spdlog::info("SoC monitor stopped"); }
//             if (tm) { tm->shutdown(); spdlog::info("Thread manager stopped"); }
//         }
//     } cleanup;

//     try {
//         //------------------------------------------------------------------
//         // 0. Metrics aggregator
//         //------------------------------------------------------------------
//         auto agg = std::make_shared<SystemMetricsAggregatorConcreteV2>();
//         if (!agg) throw std::runtime_error("Failed to create metrics aggregator");
//         agg->setCsvPath(config.value("metrics_csv", "realtime_metrics_01.csv"));
//         agg->setJsonPath(config.value("metrics_json", "realtime_metrics.ndjson"));
//         agg->setRetentionWindow(std::chrono::seconds(config.value("retention_window_sec", 120)));
//         spdlog::info("Metrics aggregator initialized");

//         //------------------------------------------------------------------
//         // 1. Queues & thread manager
//         //------------------------------------------------------------------
//         cleanup.tm = new ThreadManager();
//         auto cam2Alg = std::make_shared<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>(15);
//         auto cam2Disp = std::make_shared<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>(15);
//         auto alg2Disp = std::make_shared<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>>(15);

//         int cameraFps = config.contains("CameraConfig") ? config["CameraConfig"].value("fps", 30) : 30;
//         if (!Validation::validateQueueCapacity(60, cameraFps)) {
//             throw std::runtime_error("Invalid queue capacity");
//         }
//         spdlog::info("Queues initialized with capacity 15");
//         spdlog::info("Thread pool started with {} threads", cleanup.tm->getThreadCount());

//         //------------------------------------------------------------------
//         // 2. SoC + Power monitors
//         //------------------------------------------------------------------
//         cleanup.soc = new SoCConcrete(agg);
//         SoCConfig socCfg;
//         if (config.contains("SoCConfig")) {
//             socCfg.pollIntervalMs = config["SoCConfig"].value("pollIntervalMs", 1000);
//             socCfg.customCommand = config["SoCConfig"].value("customCommand", "tegrastats --interval 1000");
//             socCfg.exportCSV = config["SoCConfig"].value("exportCSV", false);
//             socCfg.csvPath = config["SoCConfig"].value("csvPath", "jetson_nano_tegrastats.csv");
//         }
//         if (!Validation::validateSoCConfig(socCfg)) {
//             throw std::runtime_error("Invalid SoC configuration");
//         }
//         if (!cleanup.soc->configure(socCfg)) {
//             throw std::runtime_error("SoC configuration failed");
//         }
//         int retries = 3;
//         bool socInitialized = false;
//         while (retries-- > 0 && !socInitialized) {
//             if (cleanup.soc->initializeSoC()) {
//                 socInitialized = true;
//                 spdlog::info("SoC monitor initialized");
//             } else {
//                 spdlog::warn("SoC initialization failed, retrying ({}/3)...", 3 - retries);
//                 std::this_thread::sleep_for(std::chrono::milliseconds(500));
//             }
//         }
//         if (!socInitialized) {
//             throw std::runtime_error("SoC initialization failed after retries");
//         }

//         cleanup.powerMon = new LynsynMonitorConcrete(agg, *cleanup.tm);
//         LynsynMonitorConfig pCfg;
//         if (config.contains("LynsynConfig")) {
//             pCfg.periodSampling = config["LynsynConfig"].value("periodSampling", true);
//             pCfg.outputCSV = config["LynsynConfig"].value("outputCSV", "power_raw.csv");
//             pCfg.durationSec = config["LynsynConfig"].value("durationSec", 1.0);
//             pCfg.coreMask = config["LynsynConfig"].value("coreMask", 0xF);
//         }
//         if (!Validation::validateLynsynConfig(pCfg)) {
//             throw std::runtime_error("Invalid Lynsyn configuration");
//         }
//         if (!cleanup.powerMon->configure(pCfg)) {
//             throw std::runtime_error("Lynsyn configuration failed");
//         }
//         if (!cleanup.powerMon->initialize()) {
//             throw std::runtime_error("Lynsyn monitor initialization failed");
//         }
//         cleanup.powerMon->startMonitoring();
//         spdlog::info("Power monitor started");

//         //------------------------------------------------------------------
//         // 3. Camera
//         //------------------------------------------------------------------
//         cleanup.camera = new DataConcrete(cam2Alg, cam2Disp, agg);
//         CameraConfig camCfg;
//         if (config.contains("CameraConfig")) {
//             camCfg.width = config["CameraConfig"].value("width", 640);
//             camCfg.height = config["CameraConfig"].value("height", 480);
//             camCfg.fps = config["CameraConfig"].value("fps", 30);
//             camCfg.pixelFormat = stringToPixelFormat(config["CameraConfig"].value("pixelFormat", "YUYV"));
//         } else {
//             camCfg = CameraConfig(640, 480, 30, PixelFormat::YUYV);
//         }
//         if (!Validation::validateCameraConfig(camCfg)) {
//             throw std::runtime_error("Invalid camera configuration");
//         }
//         if (!cleanup.camera->openDevice("/dev/video0")) {
//             throw std::runtime_error("Failed to open camera device");
//         }
//         if (!cleanup.camera->configure(camCfg)) {
//             throw std::runtime_error("Camera configuration failed");
//         }
//         if (!cleanup.camera->startStreaming()) {
//             throw std::runtime_error("Camera streaming failed");
//         }
//         cleanup.camera->startCapture();
//         spdlog::info("Camera started ({}x{}, {} FPS)", camCfg.width, camCfg.height, camCfg.fps);

//         //------------------------------------------------------------------
//         // 4. Algorithm
//         //------------------------------------------------------------------
//         AlgorithmConfig algCfg;
//         if (config.contains("AlgorithmConfig")) {
//             algCfg.algorithmType = stringToAlgorithmType(config["AlgorithmConfig"].value("algorithmType", "MultiThreadedInvert"));
//             algCfg.concurrencyLevel = config["AlgorithmConfig"].value("concurrencyLevel", 1);
//             algCfg.useGPU = config["AlgorithmConfig"].value("useGPU", false);
//             algCfg.blurRadius = config["AlgorithmConfig"].value("blurRadius", 5);
//             if (config["AlgorithmConfig"].contains("opticalFlowConfig")) {
//                 algCfg.opticalFlowConfig.windowSize = config["AlgorithmConfig"]["opticalFlowConfig"].value("windowSize", 15);
//                 algCfg.opticalFlowConfig.maxLevel = config["AlgorithmConfig"]["opticalFlowConfig"].value("maxLevel", 2);
//             }
//         }
//         // Line 384-385: Fix AlgorithmConcrete creation
//         if (!Validation::validateAlgorithmConfig(algCfg)) {
//             throw std::runtime_error("Invalid algorithm configuration");
//         }
//         try {
//             cleanup.algorithm = new AlgorithmConcrete(cam2Alg, alg2Disp, *cleanup.tm, agg);
//             if (!cleanup.algorithm) {
//                 throw std::runtime_error("Algorithm creation failed");
//             }
//             cleanup.algorithm->startAlgorithm();
//             spdlog::info("Algorithm started (type: {})", 
//                         config["AlgorithmConfig"].value("algorithmType", "MultiThreadedInvert"));
//         } catch (const std::exception& e) {
//             delete cleanup.algorithm;
//             cleanup.algorithm = nullptr;
//             throw std::runtime_error(std::string("Algorithm initialization failed: ") + e.what());
//         }
       
//         //------------------------------------------------------------------
//         // 5. Display
//         //------------------------------------------------------------------
//         cleanup.display = new SdlDisplayConcrete(cam2Disp, alg2Disp, agg);
//         DisplayConfig dspCfg;
//         if (config.contains("DisplayConfig")) {
//             dspCfg = DisplayConfig(
//                 config["DisplayConfig"].value("width", 640),
//                 config["DisplayConfig"].value("height", 480),
//                 config["DisplayConfig"].value("fullScreen", false),
//                 config["DisplayConfig"].value("windowTitle", "ZeroCopy Pipeline")
//             );
//         } else {
//             dspCfg = DisplayConfig(640, 480, false, "ZeroCopy Pipeline");
//         }
//         if (!Validation::validateDisplayConfig(dspCfg)) {
//             throw std::runtime_error("Invalid display configuration");
//         }
//         if (!cleanup.display->configure(dspCfg)) {
//             throw std::runtime_error("Display configuration failed");
//         }
//         if (!cleanup.display->initializeDisplay(dspCfg.width, dspCfg.height)) {
//             throw std::runtime_error("Display initialization failed");
//         }
//         spdlog::info("Display initialized ({}x{})", dspCfg.width, dspCfg.height);

//         //------------------------------------------------------------------
//         // 6. Main loop
//         //------------------------------------------------------------------
//         // Add at the top of SampleTestIntegrationCode_v3.cpp

// // ... (other includes and code remain unchanged)

// // Main loop
// auto start = std::chrono::steady_clock::now();
// size_t frameCount = 0;
// auto lastLogTime = start;
// double samplingIntervalMs = config.contains("Profiling") ? config["Profiling"].value("samplingIntervalMs", 1000) : 1000;
// double pauseDurationSec = config.contains("SystemBehavior") ? config["SystemBehavior"].value("pauseDurationSec", 2.0) : 2.0;
// bool paused = false;

// while (cleanup.display->is_Running()) {
//     auto now = std::chrono::steady_clock::now();
//     if (!paused) {
//         cleanup.display->renderAndPollEvents();
//         frameCount++;
//     }

//     // Health check
//     if (!cleanup.soc->isMonitoringActive()) {
//         spdlog::error("SoC monitor stopped unexpectedly");
//         break;
//     }
//     if (!cleanup.camera->isStreaming()) {
//         spdlog::error("Camera streaming stopped unexpectedly");
//         break;
//     }

//     // Performance logging
//     if (std::chrono::duration<double>(now - lastLogTime).count() * 1000 >= samplingIntervalMs) {
//         double fps = frameCount / std::chrono::duration<double>(now - start).count();
//         spdlog::info("Performance: FPS={:.2f}, Queues: cam2Alg={}, cam2Disp={}, alg2Disp={}",
//                      fps, cam2Alg->size(), cam2Disp->size(), alg2Disp->size());
//         // Export profiling metrics
//         if (config.contains("Profiling")) {
//             std::ofstream profFile(config["Profiling"].value("metricsOutputFile", "PerformanceMetrics.csv"), std::ios::app);
//             // Format time_point to string
//             auto now_time = std::chrono::system_clock::now();
//             auto now_c = std::chrono::system_clock::to_time_t(now_time);
//             std::stringstream ss;
//             ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
//             profFile << ss.str() << "," << fps << ","
//                      << cam2Alg->size() << "," << cam2Disp->size() << "," << alg2Disp->size() << "\n";
//         }
//         lastLogTime = now;
//     }

//     // Queue overflow mitigation
//     if (cam2Alg->isFull() || cam2Disp->isFull() || alg2Disp->isFull()) {
//         spdlog::warn("Queue overflow: cam2Alg={}, cam2Disp={}, alg2Disp={}",
//                      cam2Alg->isFull(), cam2Disp->isFull(), alg2Disp->isFull());
//         cleanup.camera->pauseCapture();
//         paused = true;
//         std::this_thread::sleep_for(std::chrono::duration<double>(pauseDurationSec));
//         cleanup.camera->resumeCapture();
//         paused = false;
//     }

//     auto elapsed = std::chrono::duration<double>(now - start).count();
//     if (elapsed > runSeconds) {
//         spdlog::info("Runtime limit ({}s) reached", runSeconds);
//         break;
//     }
//     std::this_thread::sleep_for(std::chrono::milliseconds(10));
// }
//         //------------------------------------------------------------------
//         // 7. Shutdown
//         //------------------------------------------------------------------
//         agg->exportToCSV("final_metrics.csv");
//         agg->exportToJSON("final_metrics.json");
//         spdlog::info("Metrics exported to final_metrics.csv and final_metrics.json");

//     } catch (const std::exception& e) {
//         spdlog::error("Fatal error in pipeline: {}", e.what());
//         return 1;
//     }

//     spdlog::info("Pipeline test completed successfully");
//     return 0;
// }