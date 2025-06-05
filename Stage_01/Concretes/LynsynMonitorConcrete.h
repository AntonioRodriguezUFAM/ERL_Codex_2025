
// LynsynMonitorConcrete.h

#pragma once

#include "../Interfaces/ILynsynMonitor.h"

#include "../SharedStructures/ThreadManager.h"

//#include "../SharedStructures/LynsynMonitorConfig.h"
//#include "../SharedStructures/PowerStats.h"
#include "../Interfaces/ISystemMetricsAggregator.h"


//#include "../SharedStructures/SystemMetricsAggregator.h"      // You should define the aggregator
//#include "../SharedStructures/PowerStats.h"                     // The structure shown above


#include "../Interfaces/ISystemMetricsAggregator.h"
#include "../SharedStructures/allModulesStatcs.h"               // The structure shown above

#include <thread>

#include <functional>
#include <memory>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <sstream>
#include <vector>


#include <iostream>
#include <fstream>
#include <spdlog/spdlog.h>

extern "C" {
    #include "../Includes/lynsyn.h" // The C library for Lynsyn
}

/**
 * @class LynsynMonitorConcrete
 * @brief Concrete class implementing ILynsynMonitor. Wraps around Lynsyn library calls
 *        to measure current and voltage on the Jetson Nano (or other boards).
 */
class LynsynMonitorConcrete : public ILynsynMonitor {
public:
    //LynsynMonitorConcrete();
    //explicit LynsynMonitorConcrete(std::shared_ptr<ISystemMetricsAggregator> aggregator);

    

    // Constructor with optional aggregator threa
    explicit LynsynMonitorConcrete(std::shared_ptr<ISystemMetricsAggregator> aggregator ,ThreadManager& threadManager);
  

    ~LynsynMonitorConcrete() override;

    bool configure(const LynsynMonitorConfig& config) override;
    bool initialize() override;
    void startMonitoring() override;
    void stop() override;

    void setSampleCallback(std::function<void(const LynsynSample&)> callback) override;
    void setErrorCallback(std::function<void(const std::string&)>) override;
    

private:
    /**
     * @brief Worker thread function that continuously reads from Lynsyn
     *        and passes samples to the user callback (push-based design).
     */
    void monitoringThreadLoop();

    /**
     * @brief Called internally to log or handle errors.
     */
    void reportError(const std::string& msg);

    PowerStats convertLynsynSampleToPowerStats(const LynsynSample& sample);

private:
    LynsynMonitorConfig config_;

     // Optional callbacks
    std::function<void(const LynsynSample&)> sampleCallback_; // Added missing member
    std::function<void(const std::string&)> errorCallback_;

    // Lynsyn library state
    std::thread monitoringThread_;


    // Lynsyn state
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> threadShouldExit_{false};
   


    // For simple CSV logging if user sets config_.outputCSV
    std::ofstream outputFile_;

  // DataConcrete to support SystemMetricsAggregatorImpl injection and push camera metrics in real-time using the aggregator
  std::shared_ptr<ISystemMetricsAggregator> metricAggregator_;

};

// ------------------------------------ Constructor and Destructor ----------------------------------------------//
// -------------------------------------- Implementation --------------------------------------------------------//
// --------------------------------------------------------------------------------------------------------------//



// LynsynMonitorConcrete - Constructor with optional aggregator thread
inline LynsynMonitorConcrete::LynsynMonitorConcrete(std::shared_ptr<ISystemMetricsAggregator> aggregator, ThreadManager& threadManager)
    : initialized_(false)
    , running_(false)
    , threadShouldExit_(false)
    , metricAggregator_(std::move(aggregator)) 
        {
        // Constructor implementation
        // You can initialize other members here if needed
        // For example, you might want to set up the aggregator or other resources
        // metricAggregator_ = std::move(aggregator);
        // Initialize other members if needed
        // initialized_ = false;
        // running_ = false;
        // threadShouldExit_ = false;
        // monitoringThread_ = std::thread();
        // outputFile_ = std::ofstream();
        // errorCallback_ = nullptr; // or some default function

        // MODIFIED: Mark 'threadManager' as used
        (void)threadManager;  // Suppresses unused parameter warning

        }


        
inline LynsynMonitorConcrete::~LynsynMonitorConcrete() {
    stop(); // Ensure resources are released
}

inline bool LynsynMonitorConcrete::configure(const LynsynMonitorConfig& config) {
    config_ = config;
    return true; // if config is valid
}

inline bool LynsynMonitorConcrete::initialize() {
    // Example: call lynsyn_init() from the C library
    if (!lynsyn_init()) {
        reportError("Failed to init Lynsyn library. Device not found or in use?");
        return false;
    }
    // Possibly do more setup
    initialized_ = true;
    return true;
}

inline void LynsynMonitorConcrete::startMonitoring() {
    if (!initialized_) {
        reportError("LynsynMonitorConcrete::startMonitoring called but not initialized.");
        return;
    }

    if (running_) {
        reportError("LynsynMonitorConcrete::startMonitoring called but already running.");
        return;
    }

    // If user wants CSV output
    if (!config_.outputCSV.empty()) {
        outputFile_.open(config_.outputCSV);
        if (!outputFile_.is_open()) {
            reportError("Could not open output file for Lynsyn CSV logging: " + config_.outputCSV);
        } else {
            // Write CSV header
            outputFile_ << "Time,Core0_PC,Core1_PC,Core2_PC,Core3_PC,"
                        << "Current0,Current1,Current2,Voltage0,Voltage1,Voltage2\n";
        }
    }

    // Start the sampling now:
    if (config_.periodSampling) {
        // time-based sampling
        lynsyn_startPeriodSampling(config_.durationSec, config_.coreMask);
    } else {
        // e.g. breakpoint-based (or combined)
        if (config_.startBreakpoint != 0 && config_.endBreakpoint != 0) {
            lynsyn_startBpSampling(config_.startBreakpoint, config_.endBreakpoint, config_.coreMask);
        } else {
            // fallback
            lynsyn_startPeriodSampling(config_.durationSec, config_.coreMask);
        }
    }

    // Start background thread
    threadShouldExit_ = false;
    monitoringThread_ = std::thread(&LynsynMonitorConcrete::monitoringThreadLoop, this);
    running_ = true;
}

inline void LynsynMonitorConcrete::stop() {
    // Signal the thread to exit
    threadShouldExit_ = true;
    if (monitoringThread_.joinable()) {
        monitoringThread_.join();
    }

    // close Lynsyn
    lynsyn_release();
    initialized_ = false;
    running_ = false;

    if (outputFile_.is_open()) {
        outputFile_.close();
    }
}


// Helper function to convert LynsynSample to PowerStats
inline PowerStats LynsynMonitorConcrete::convertLynsynSampleToPowerStats(const LynsynSample& sample) {
    PowerStats stats;
    stats.timestamp = std::chrono::system_clock::now(); // Use sample's timestamp if available

    for (int i = 0; i < LYNSYN_MAX_SENSORS; ++i) {
       // stats.currents.push_back(sample.current[i]);
       // stats.voltages.push_back(sample.voltage[i]);

        stats.currents.push_back(static_cast<double>(sample.current[i]));
        stats.voltages.push_back(static_cast<double>(sample.voltage[i]));
      //  std::cout << "Current: " << sample.current[i] << "A, Voltage: " << sample.voltage[i] << "V" << std::endl;
    }

    return stats;
}


// Final implementation of the monitoring thread loop
// This function will be called in a separate thread
// and will continuously read samples from the Lynsyn device
inline void LynsynMonitorConcrete::monitoringThreadLoop() {
    spdlog::debug("[LynsynMonitorConcrete] monitoringThreadLoop started.");

    while (!threadShouldExit_) {
        LynsynSample sample;
        if (!lynsyn_getNextSample(&sample)) {
            spdlog::warn("[LynsynMonitorConcrete] Failed to read Lynsyn sample. Exiting loop.");
            break;
        }

        // Check if the sampling session has ended
        if (sample.flags & SAMPLE_FLAG_HALTED) {
            spdlog::info("[LynsynMonitorConcrete] Sampling halted. Exiting loop.");
            break;
        }
        // Check if the sample is valid
        // spdlog::info("[LynsynMonitorConcrete] Sample received:time={} cycles, flags={},Current0={} mA,Voltage0={} V ",
        //                     sample.time,
        //                     sample.flags,
        //                     sample.current[0],
        //                     sample.voltage[0]);

                

         //Convert to PowerStats
        PowerStats stats = convertToPowerStats(sample);

        // Push metrics to aggregator
        // Push into SystemMetricsSnapshot
        if (metricAggregator_) {
            metricAggregator_->pushMetrics(stats.timestamp, [&](SystemMetricsSnapshot& snap) {
                snap.powerStats = stats; //  This copies all PowerStats data into the snapshot
            });

        }

        // Optional callback
               // Invoke external callback if set
        if (sampleCallback_) {
            sampleCallback_(sample);
        }

        // Write to CSV if enabled
        if (outputFile_.is_open()) {
            // Write timestamp
            double seconds = lynsyn_cyclesToSeconds(sample.time);
            outputFile_ << seconds;

            // Write program counters
            for (int i = 0; i < 4; ++i) {
                outputFile_ << "," << sample.pc[i];
            }

            // Write current and voltage values
            for (size_t i = 0; i < stats.sensorCount(); ++i) {
                outputFile_ << "," << stats.currents[i] << "," << stats.voltages[i];
            }
            // Write power values
            outputFile_ << "," << stats.totalPower() << "," << stats.averagePower() << "\n";
        }
    }

    spdlog::info("[LynsynMonitorConcrete] monitoringThreadLoop exiting cleanly.");
}



inline void LynsynMonitorConcrete::setSampleCallback(std::function<void(const LynsynSample&)> callback) {
    sampleCallback_ = std::move(callback);
}

inline void LynsynMonitorConcrete::setErrorCallback(std::function<void(const std::string&)> cb) {
    errorCallback_ = std::move(cb);
}

inline void LynsynMonitorConcrete::reportError(const std::string& msg) {
    if (errorCallback_) {
        errorCallback_(msg);
    } else {
        // fallback
        fprintf(stderr, "[LynsynMonitorConcrete] Error: %s\n", msg.c_str());
    }
}
