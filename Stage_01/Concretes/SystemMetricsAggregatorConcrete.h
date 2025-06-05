// SystemMetricsAggregatorConcrete.h

//____________________________________________________________________
// FINAL VERSION
//______________________________________________________________________
// Final version system aggregation 


#pragma once

#include "../Interfaces/ISystemMetricsAggregator.h"
#include "../SharedStructures/allModulesStatcs.h"
#include "../SharedStructures/AggregatorConfig.h" // This should contain the definition of SystemMetricsSnapshot
#include <mutex>
#include <vector>
#include "../nlohmann/json.hpp"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <iomanip>

using json = nlohmann::json;

/**
 * @class SystemMetricsAggregatorConcrete
 * @brief Concrete implementation of the ISystemMetricsAggregator interface.
 * This class provides a thread-safe, efficient way to collect, aggregate, and export system metrics.
 */
class SystemMetricsAggregatorConcrete : public ISystemMetricsAggregator {
public:
    SystemMetricsAggregatorConcrete() = default;
    ~SystemMetricsAggregatorConcrete() override = default;

    void pushMetrics(
        const std::chrono::system_clock::time_point& timestamp,
        std::function<void(SystemMetricsSnapshot&)> updateFn) override;

    SystemMetricsSnapshot getLatestSnapshot() const override;
    std::vector<SystemMetricsSnapshot> getAllSnapshots() const override;
    SystemMetricsSnapshot getAggregatedAt(std::chrono::system_clock::time_point ts) const override;

    void pushCameraStats(const CameraStats& stats) override;
    void pushAlgorithmStats(const AlgorithmStats& stats) override;
    void pushDisplayStats(const DisplayStats& stats) override;
    void pushSoCStats(const JetsonNanoInfo& stats) override;
    void pushPowerStats(const PowerStats& stats) override;

    void exportToCSV(const std::string& filePath) override;
    void exportToJSON(const std::string& filePath) override;

    void setRetentionWindow(std::chrono::seconds window) { retentionWindow_ = window; }

private:
    template<typename T>
    void pushMetricsHelper(const T& data, 
        std::function<void(SystemMetricsSnapshot&, const T&)> updateFn);

    template<typename T>
    T aggregateField(std::function<T(const SystemMetricsSnapshot&)> extractor,
        std::chrono::system_clock::time_point ts) const;

        
    void appendSnapshotToCSV(const SystemMetricsSnapshot& snap, const std::string& filePath);
    void appendSnapshotToJSON(const SystemMetricsSnapshot& snap, const std::string& filePath);

    std::string formatTimestamp(const std::chrono::system_clock::time_point& tp) const;
    json snapshotToJson(const SystemMetricsSnapshot& snap) const;

    void enforceRetentionPolicy();

    mutable std::mutex mutex_;
    std::vector<SystemMetricsSnapshot> snapshots_;
    std::chrono::seconds retentionWindow_;


    /**
     * @brief Retrieves the configuration settings for the aggregator.
     * @return The AggregatorConfig structure containing configuration settings.
     */     
    std::shared_ptr<ISystemMetricsAggregator> agg = std::make_shared<SystemMetricsAggregatorConcrete>();

};


//====================================================================

// #pragma once

// #include <iomanip>

// #include "../Interfaces/ISystemMetricsAggregator.h"
// #include "../SharedStructures/allModulesStatcs.h" // This should contain the definition of SystemMetricsSnapshot
 
// #include "../nlohmann/json.hpp" // Export to JSON // Include the nlohmann/json.hpp header for JSON serialization
// using json = nlohmann::json;
// #include <fstream>
// #include <mutex>
// #include <vector>
// #include <spdlog/spdlog.h>


// class SystemMetricsAggregatorConcrete : public ISystemMetricsAggregator {
//     public:
//         SystemMetricsAggregatorConcrete() = default;
//         ~SystemMetricsAggregatorConcrete() override = default;
       
        
//         // Interface implementations
//         void pushMetrics(
//             const std::chrono::system_clock::time_point& timestamp,
//             std::function<void(SystemMetricsSnapshot&)> updateFn) override;

//         SystemMetricsSnapshot getLatestSnapshot() const override;
//         std::vector<SystemMetricsSnapshot> getAllSnapshots()const override;
//         SystemMetricsSnapshot getAggregatedAt(std::chrono::system_clock::time_point ts) const;
        

//         // Metric collection interface

//         void pushCameraStats(const CameraStats& stats) override;
//         void pushAlgorithmStats(const AlgorithmStats& stats) override;
//         void pushDisplayStats(const DisplayStats& stats) override;
//         void pushSoCStats(const JetsonNanoInfo& stats) override;
//         void pushPowerStats(const PowerStats& stats) override;


//         // Export to CSV and JSON
//         void exportToCSV(const std::string& filePath) override;
//         void exportToJSON(const std::string& filePath) override;
      
        
// private:
//         // Template helper for metric collection
//         template<typename T>
//         void pushMetricsHelper(const T& data, 
//                             std::function<void(SystemMetricsSnapshot&, const T&)> updater) {
//             pushMetrics(data.timestamp, [&](SystemMetricsSnapshot& snap) {
//                 updater(snap, data);
//             });
//         }
//         // ---- Private members ----
//         // Aggregation implementation
//         template<typename T>
//         T aggregateField(std::function<double(const T&)> extractor,
//                         std::chrono::system_clock::time_point ts) const {
//             T result{};
//             size_t count = 0;
            
//             std::lock_guard<std::mutex> lock(mutex_);
//             for(const auto& snap : snapshots_) {
//                 if(snap.timestamp <= ts) {
//                     result += extractor(snap);
//                     ++count;
//                 }
//             }
//             return count > 0 ? result / count : T{};
//         }
        


//       // Append to CSV and JSON
//      void appendSnapshotToCSV(const SystemMetricsSnapshot& snap, const std::string& filePath);
//     void appendSnapshotToJSON(const SystemMetricsSnapshot& snap, const std::string& filePath);
    
//     mutable std::mutex mutex_;
//     std::vector<SystemMetricsSnapshot> snapshots_;

//         // Helper function to format timestamps
//         // This function formats a timestamp into a human-readable string
//     std::string formatTimestamp(const std::chrono::system_clock::time_point& tp) const {
//             auto timeT = std::chrono::system_clock::to_time_t(tp);
//             struct tm buf;
//             localtime_r(&timeT, &buf);
//             std::ostringstream ss;
//             ss << std::put_time(&buf, "%Y-%m-%d %H:%M:%S");
//             return ss.str();
//         }
// // Helper function to convert a snapshot to JSON
//     // This function converts a SystemMetricsSnapshot to a JSON object
//     json snapshotToJson(const SystemMetricsSnapshot& snap) const {
//         json j;
//         j["timestamp"] = formatTimestamp(snap.timestamp);
        
//         // Camera metrics
//         j["camera"] = {
//             {"frame_number", snap.cameraStats.frameNumber},
//             {"fps", snap.cameraStats.fps},
//             {"width", snap.cameraStats.frameWidth},
//             {"height", snap.cameraStats.frameHeight},
//             {"size_bytes", snap.cameraStats.frameSize}
//         };

//         // Algorithm metrics
//         j["algorithm"] = {
//             {"inference_time_ms", snap.algorithmStats.inferenceTimeMs},
//             {"confidence", snap.algorithmStats.confidenceScore},
//             {"fps", snap.algorithmStats.fps_},
//             {"avg_processing_time", snap.algorithmStats.avgProcTime_},
//             {"total_processing_time", snap.algorithmStats.totalProcTime_}
//         };

//         // Display metrics
//         j["display"] = {
//             {"latency_ms", snap.displayStats.latencyMs},
//             {"dropped_frames", snap.displayStats.droppedFrames},
//             {"render_time_ms", snap.displayStats.renderTimeMs}
//         };

//         // SoC metrics
//         j["soc"] = {
//             {"ram", {
//                 {"used_mb", snap.socInfo.RAM_In_Use_MB},
//                 {"total_mb", snap.socInfo.Total_RAM_MB},
//                 {"cached_mb", snap.socInfo.Cached_MB}
//             }},
//             {"swap", {
//                 {"used_mb", snap.socInfo.SWAP_In_Use_MB},
//                 {"total_mb", snap.socInfo.Total_SWAP_MB}
//             }},
//             // ... other SoC fields
//         };

//         // Power metrics
//         json powerJson;
//         powerJson["total"] = snap.powerStats.totalPower();
//         for(size_t i = 0; i < snap.powerStats.sensorCount(); ++i) {
//             powerJson["sensors"][std::to_string(i)] = {
//                 {"power", snap.powerStats.sensorPower(i)},
//                 {"voltage", snap.powerStats.voltages[i]},
//                 {"current", snap.powerStats.currents[i]}
//             };
//         }
//         j["power"] = powerJson;

//         return j;
//     }

        
//     };
    

// --------------------------------------- Implementation -------------------------------------------------------//
// ------------------------------------ Constructor and Destructor ----------------------------------------------//
// -------------------------------------- Implementation --------------------------------------------------------//
// --------------------------------------------------------------------------------------------------------------//

// Constructor
void SystemMetricsAggregatorConcrete::pushMetrics(
    const std::chrono::system_clock::time_point& timestamp,
    std::function<void(SystemMetricsSnapshot&)> updateFn) 
{
   // std::lock_guard<std::mutex> lock(mutex_); // Ensure mutex_ is defined
    // Create a new snapshot with the given timestamp
    SystemMetricsSnapshot snapshot;
    snapshot.timestamp = timestamp;
    updateFn(snapshot);

    // Lock the mutex to safely add the snapshot to the collection
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshots_.push_back(snapshot);
        // Enforce retention policy to keep only relevant snapshots
        enforceRetentionPolicy();
    }

    // Append the snapshot to real-time files without holding the lock
    appendSnapshotToCSV(snapshot, "realtime_metrics.csv");
    appendSnapshotToJSON(snapshot, "realtime_metrics.ndjson");
}

template<typename T>
void SystemMetricsAggregatorConcrete::pushMetricsHelper(
    const T& data, 
    std::function<void(SystemMetricsSnapshot&, const T&)> updateFn) 
{
    // Use the main pushMetrics function with a lambda that updates the snapshot
    //pushMetrics(data.timestamp, [&](auto& snap) {
    pushMetrics(data.timestamp, [data, updateFn](auto& snap) {
        updateFn(snap, data);
    });
}

SystemMetricsSnapshot SystemMetricsAggregatorConcrete::getLatestSnapshot() const 
{
    // Return the latest snapshot or a default-constructed one if none exist
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshots_.empty() ? SystemMetricsSnapshot{} : snapshots_.back();
}

std::vector<SystemMetricsSnapshot> SystemMetricsAggregatorConcrete::getAllSnapshots() const 
{
    // Return a copy of all snapshots
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshots_;
}

SystemMetricsSnapshot SystemMetricsAggregatorConcrete::getAggregatedAt(
    std::chrono::system_clock::time_point ts) const 
{
    SystemMetricsSnapshot aggregated;

    // Find the snapshot closest to the given timestamp
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::lower_bound(snapshots_.begin(), snapshots_.end(), ts,
        [](const auto& snap, auto time) { return snap.timestamp < time; });

    if (it != snapshots_.end()) {
        aggregated = *it;
    }

    return aggregated;
}
inline void SystemMetricsAggregatorConcrete::pushCameraStats(const CameraStats& stats) {
    pushMetricsHelper<CameraStats>(stats, 
        [](SystemMetricsSnapshot& snap, const CameraStats& data) {
            snap.cameraStats = data;
        });
}

inline void SystemMetricsAggregatorConcrete::pushAlgorithmStats(const AlgorithmStats& stats) {
    pushMetricsHelper<AlgorithmStats>(stats,
        [](SystemMetricsSnapshot& snap, const AlgorithmStats& data) {
            snap.algorithmStats = data;
        });
}

inline void SystemMetricsAggregatorConcrete::pushDisplayStats(const DisplayStats& stats) {
    pushMetricsHelper<DisplayStats>(stats,
        [](SystemMetricsSnapshot& snap, const DisplayStats& data) {
            snap.displayStats = data;
        });
}

inline void SystemMetricsAggregatorConcrete::pushSoCStats(const JetsonNanoInfo& stats) {
    pushMetricsHelper<JetsonNanoInfo>(stats,
        [](SystemMetricsSnapshot& snap, const JetsonNanoInfo& data) {
            snap.socInfo = data;
        });
}

inline void SystemMetricsAggregatorConcrete::pushPowerStats(const PowerStats& stats) {
    pushMetricsHelper<PowerStats>(stats,
        [](SystemMetricsSnapshot& snap, const PowerStats& data) {
            snap.powerStats = data;
        });
}

//==================================================================================================
void SystemMetricsAggregatorConcrete::exportToCSV(const std::string& filePath) {
    // Create a copy of the snapshots vector to avoid blocking other operations during file I/O
    std::vector<SystemMetricsSnapshot> copy;
    {
        std::lock_guard<std::mutex> lock(mutex_); // Lock the mutex to safely copy the snapshots
        copy = snapshots_;
    }

    try {
        std::ofstream file(filePath); // Open the CSV file for writing
        if (!file) throw std::runtime_error("Failed to open CSV file"); // Check if the file was opened successfully

        // Write the header row with all relevant metric field names
        file << "Timestamp,"
             << "CameraFrameNumber,CameraFPS,CameraWidth,CameraHeight,CameraFrameSize,"
             << "AlgorithmInferenceTimeMs,AlgorithmConfidence,AlgorithmFPS,AlgorithmAvgProcTimeMs,AlgorithmTotalProcTime,"
             << "DisplayLatencyMs,DroppedFrames,RenderTimeMs,"
             << "RAM_Used_MB,RAM_Total_MB,SWAP_Used_MB,SWAP_Total_MB,Cached_MB,"
             << "CPU1_Utilization_Percent,CPU1_Frequency_MHz,CPU2_Utilization_Percent,CPU2_Frequency_MHz,"
             << "CPU3_Utilization_Percent,CPU3_Frequency_MHz,CPU4_Utilization_Percent,CPU4_Frequency_MHz,"
             << "EMC_Frequency_Percent,GR3D_Frequency_Percent,"
             << "PLL_Temp_C,CPU_Temp_C,PMIC_Temp_C,GPU_Temp_C,AO_Temp_C,Thermal_Temp_C,"
             << "PowerTotalW,PowerAverageW,PowerSensor0W,PowerSensor0V,PowerSensor0A,PowerSensor1W,PowerSensor1V,PowerSensor1A,PowerSensor2W,PowerSensor2V,PowerSensor2A,PowerSensor3W,PowerSensor3V,PowerSensor3A\n";

        // Iterate through each snapshot and write its data to the CSV file
        for (const auto& snap : copy) {
            file << formatTimestamp(snap.timestamp) << "," // Write the formatted timestamp
                 << snap.cameraStats.frameNumber << "," << snap.cameraStats.fps << "," // Camera metrics
                 << snap.cameraStats.frameWidth << "," << snap.cameraStats.frameHeight << ","
                 << snap.cameraStats.frameSize << ","
                 << snap.algorithmStats.inferenceTimeMs << "," << snap.algorithmStats.confidenceScore << "," // Algorithm metrics
                 << snap.algorithmStats.fps_ << "," << snap.algorithmStats.avgProcTime_ << ","
                 << snap.algorithmStats.totalProcTime_ << ","
                 << snap.displayStats.latencyMs << "," << snap.displayStats.droppedFrames << "," // Display metrics
                 << snap.displayStats.renderTimeMs << ","
                 << snap.socInfo.RAM_In_Use_MB << "," << snap.socInfo.Total_RAM_MB << "," // SoC metrics
                 << snap.socInfo.SWAP_In_Use_MB << "," << snap.socInfo.Total_SWAP_MB << ","
                 << snap.socInfo.Cached_MB << ","
                 << snap.socInfo.CPU1_Utilization_Percent << "," << snap.socInfo.CPU1_Frequency_MHz << ","
                 << snap.socInfo.CPU2_Utilization_Percent << "," << snap.socInfo.CPU2_Frequency_MHz << ","
                 << snap.socInfo.CPU3_Utilization_Percent << "," << snap.socInfo.CPU3_Frequency_MHz << ","
                 << snap.socInfo.CPU4_Utilization_Percent << "," << snap.socInfo.CPU4_Frequency_MHz << ","
                 << snap.socInfo.EMC_Frequency_Percent << "," << snap.socInfo.GR3D_Frequency_Percent << ","
                 << snap.socInfo.PLL_Temperature_C << "," << snap.socInfo.CPU_Temperature_C << "," // Temperature metrics
                 << snap.socInfo.PMIC_Temperature_C << "," << snap.socInfo.GPU_Temperature_C << ","
                 << snap.socInfo.AO_Temperature_C << "," << snap.socInfo.Thermal_Temperature_C << ","
                 << snap.powerStats.totalPower() << ","
                 << snap.powerStats.averagePower() << ","
                 << snap.powerStats.sensorPower(0) << "," << snap.powerStats.voltages[0] << "," << snap.powerStats.currents[0] << ","
                 << snap.powerStats.sensorPower(1) << "," << snap.powerStats.voltages[1] << "," << snap.powerStats.currents[1] << ","
                 << snap.powerStats.sensorPower(2) << "," << snap.powerStats.voltages[2] << "," << snap.powerStats.currents[2] << ","
                 << snap.powerStats.sensorPower(3) << "," << snap.powerStats.voltages[3] << "," << snap.powerStats.currents[3]
                 << "\n";
        }
    } catch (const std::exception& e) {
        spdlog::error("CSV export failed: {}", e.what()); // Log any exceptions that occur during export
        throw; // Re-throw the exception to allow the caller to handle it if needed
    }
}

void SystemMetricsAggregatorConcrete::exportToJSON(const std::string& filePath) {
    // Export all snapshots to a JSON file
    std::vector<SystemMetricsSnapshot> copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        copy = snapshots_;
    }

    try {
        json j;
        for (const auto& snap : copy) {
            j.push_back(snapshotToJson(snap));
        }

        std::ofstream file(filePath);
        if (!file) throw std::runtime_error("Failed to open JSON file");
        file << j.dump(4);
    }
    catch (const std::exception& e) {
        spdlog::error("JSON export failed: {}", e.what());
        throw;
    }
}


void SystemMetricsAggregatorConcrete::appendSnapshotToCSV(
    const SystemMetricsSnapshot& snap, 
    const std::string& filePath) 
{
    // Append a single snapshot to a CSV file
    std::ofstream file(filePath, std::ios::app);
    if (!file) return;

    // Write header if the file is new
    if (file.tellp() == 0) {
        file << "Timestamp,"
             << "CameraFrameNumber,CameraFPS,CameraWidth,CameraHeight,CameraFrameSize,"
             << "AlgorithmInferenceTimeMs,AlgorithmConfidence,AlgorithmFPS,AlgorithmAvgProcTimeMs,AlgorithmTotalProcTime,"
             << "DisplayLatencyMs,DroppedFrames,RenderTimeMs,"
             << "RAM_Used_MB,RAM_Total_MB,SWAP_Used_MB,SWAP_Total_MB,Cached_MB,"
             << "CPU1_Utilization_Percent,CPU1_Frequency_MHz,CPU2_Utilization_Percent,CPU2_Frequency_MHz,"
             << "CPU3_Utilization_Percent,CPU3_Frequency_MHz,CPU4_Utilization_Percent,CPU4_Frequency_MHz,"
             << "EMC_Frequency_Percent,GR3D_Frequency_Percent,"
             << "PLL_Temp_C,CPU_Temp_C,PMIC_Temp_C,GPU_Temp_C,AO_Temp_C,Thermal_Temp_C,"
             << "PowerTotalW,PowerAverageW,PowerSensor0W,PowerSensor0V,PowerSensor0A,PowerSensor1W,PowerSensor1V,PowerSensor1A,PowerSensor2W,PowerSensor2V,PowerSensor2A,PowerSensor3W,PowerSensor3V,PowerSensor3A\n";
    }

    // Write the snapshot data to the CSV file
    file << formatTimestamp(snap.timestamp) << "," // Write the formatted timestamp
         << snap.cameraStats.frameNumber << "," << snap.cameraStats.fps << "," // Camera metrics
         << snap.cameraStats.frameWidth << "," << snap.cameraStats.frameHeight << ","
         << snap.cameraStats.frameSize << ","
         << snap.algorithmStats.inferenceTimeMs << "," << snap.algorithmStats.confidenceScore << "," // Algorithm metrics
         << snap.algorithmStats.fps_ << "," << snap.algorithmStats.avgProcTime_ << ","
         << snap.algorithmStats.totalProcTime_ << ","
         << snap.displayStats.latencyMs << "," << snap.displayStats.droppedFrames << "," // Display metrics
         << snap.displayStats.renderTimeMs << ","
         << snap.socInfo.RAM_In_Use_MB << "," << snap.socInfo.Total_RAM_MB << "," // SoC metrics
         << snap.socInfo.SWAP_In_Use_MB << "," << snap.socInfo.Total_SWAP_MB << ","
         << snap.socInfo.Cached_MB << ","
         << snap.socInfo.CPU1_Utilization_Percent << "," << snap.socInfo.CPU1_Frequency_MHz << ","
         << snap.socInfo.CPU2_Utilization_Percent << "," << snap.socInfo.CPU2_Frequency_MHz << ","
         << snap.socInfo.CPU3_Utilization_Percent << "," << snap.socInfo.CPU3_Frequency_MHz << ","
         << snap.socInfo.CPU4_Utilization_Percent << "," << snap.socInfo.CPU4_Frequency_MHz << ","
         << snap.socInfo.EMC_Frequency_Percent << "," << snap.socInfo.GR3D_Frequency_Percent << ","
         << snap.socInfo.PLL_Temperature_C << "," << snap.socInfo.CPU_Temperature_C << "," // Temperature metrics
         << snap.socInfo.PMIC_Temperature_C << "," << snap.socInfo.GPU_Temperature_C << ","
         << snap.socInfo.AO_Temperature_C << "," << snap.socInfo.Thermal_Temperature_C << ","
         << snap.powerStats.totalPower() << ","
         << snap.powerStats.averagePower() << ","
         << snap.powerStats.sensorPower(0) << "," << snap.powerStats.voltages[0] << "," << snap.powerStats.currents[0] << ","
         << snap.powerStats.sensorPower(1) << "," << snap.powerStats.voltages[1] << "," << snap.powerStats.currents[1] << ","
         << snap.powerStats.sensorPower(2) << "," << snap.powerStats.voltages[2] << "," << snap.powerStats.currents[2] << ","
         << snap.powerStats.sensorPower(3) << "," << snap.powerStats.voltages[3] << "," << snap.powerStats.currents[3]
         << "\n";

}

void SystemMetricsAggregatorConcrete::appendSnapshotToJSON(
    const SystemMetricsSnapshot& snap,
    const std::string& filePath) 
{
    // Append a single snapshot to a JSON file
    std::ofstream file(filePath, std::ios::app);
    if (!file) return;

    file << snapshotToJson(snap).dump() << "\n";
}

std::string SystemMetricsAggregatorConcrete::formatTimestamp(
    const std::chrono::system_clock::time_point& tp) const 
{
    // Convert a timestamp to a human-readable string with milliseconds
    auto timeT = std::chrono::system_clock::to_time_t(tp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        tp.time_since_epoch()) % 1000;

    struct tm buf;
    localtime_r(&timeT, &buf);
    
    std::ostringstream ss;
    ss << std::put_time(&buf, "%Y-%m-%d %H:%M:%S") << "." 
       << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

json SystemMetricsAggregatorConcrete::snapshotToJson(
    const SystemMetricsSnapshot& snap) const 
{
    // Convert a snapshot to a JSON object
    json j;
    j["timestamp"] = formatTimestamp(snap.timestamp);

    j["camera"] = {
        {"frame_number", snap.cameraStats.frameNumber},
        {"fps", snap.cameraStats.fps},
        {"width", snap.cameraStats.frameWidth},
        {"height", snap.cameraStats.frameHeight}
    };

    j["algorithm"] = {
        {"inference_time_ms", snap.algorithmStats.inferenceTimeMs},
        {"confidence", snap.algorithmStats.confidenceScore}
    };

    j["display"] = {
        {"latency_ms", snap.displayStats.latencyMs},
        {"dropped_frames", snap.displayStats.droppedFrames}
    };

    j["soc"] = {
        {"ram_used_mb", snap.socInfo.RAM_In_Use_MB},
        {"ram_total_mb", snap.socInfo.Total_RAM_MB},
        {"cpu1_utilization", snap.socInfo.CPU1_Utilization_Percent}
    };

    j["power"] = {
        {"total", snap.powerStats.totalPower()},
        {"average", snap.powerStats.averagePower()}
    };

    return j;
}

void SystemMetricsAggregatorConcrete::enforceRetentionPolicy() {
    // Remove snapshots older than the retention window
    if (retentionWindow_ <= std::chrono::seconds(0)) return;

    auto cutoff = std::chrono::system_clock::now() - retentionWindow_;
    auto it = std::lower_bound(snapshots_.begin(), snapshots_.end(), cutoff,
        [](const auto& snap, auto time) { return snap.timestamp < time; });

    if (it != snapshots_.begin()) {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshots_.erase(snapshots_.begin(), it);
    }
}






// //==================================================================================================

// //SystemMetricsAggregatorConcrete::~SystemMetricsAggregatorConcrete() {}
// // --------------------------------------------------------------------------------------------------------------//
// inline void SystemMetricsAggregatorConcrete::pushMetrics(
//     const std::chrono::system_clock::time_point& timestamp,
//     std::function<void(SystemMetricsSnapshot&)> updateFn) {
//     SystemMetricsSnapshot snapshot;
//     snapshot.timestamp = timestamp;
//     updateFn(snapshot);

//     std::lock_guard<std::mutex> lock(mutex_);
//     snapshots_.push_back(snapshot);
// }



// // Get the latest snapshot
// inline SystemMetricsSnapshot SystemMetricsAggregatorConcrete::getLatestSnapshot() const {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return snapshots_.empty() ? SystemMetricsSnapshot{} : snapshots_.back();
// }

// // Get all snapshots
// inline std::vector<SystemMetricsSnapshot> SystemMetricsAggregatorConcrete::getAllSnapshots() const {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return snapshots_;
// }

// // Get aggregated metrics at a specific timestamp
// // This function aggregates the metrics at a specific timestamp
// inline SystemMetricsSnapshot SystemMetricsAggregatorConcrete::getAggregatedAt(
//     std::chrono::system_clock::time_point ts) const {
//     SystemMetricsSnapshot aggregated;
    
//     std::lock_guard<std::mutex> lock(mutex_);
//     auto it = std::lower_bound(snapshots_.begin(), snapshots_.end(), ts,
//         [](const auto& snap, auto time) { return snap.timestamp < time; });

//     if(it != snapshots_.end()) {
//         aggregated = *it;
//         // Add custom aggregation logic here
//     }
//     return aggregated;
// }

// // --------------------------------------------------------------------------------------------------------------//


// // Metric collection implementations using helper
// inline void SystemMetricsAggregatorConcrete::pushCameraStats(const CameraStats& stats) {
//     pushMetricsHelper<CameraStats>(stats, 
//         [](SystemMetricsSnapshot& snap, const CameraStats& data) {
//             snap.cameraStats = data;
//         });
// }

// inline void SystemMetricsAggregatorConcrete::pushAlgorithmStats(const AlgorithmStats& stats) {
//     pushMetricsHelper<AlgorithmStats>(stats,
//         [](SystemMetricsSnapshot& snap, const AlgorithmStats& data) {
//             snap.algorithmStats = data;
//         });
// }

// inline void SystemMetricsAggregatorConcrete::pushDisplayStats(const DisplayStats& stats) {
//     pushMetricsHelper<DisplayStats>(stats,
//         [](SystemMetricsSnapshot& snap, const DisplayStats& data) {
//             snap.displayStats = data;
//         });
// }

// inline void SystemMetricsAggregatorConcrete::pushSoCStats(const JetsonNanoInfo& stats) {
//     pushMetricsHelper<JetsonNanoInfo>(stats,
//         [](SystemMetricsSnapshot& snap, const JetsonNanoInfo& data) {
//             snap.socInfo = data;
//         });
// }

// inline void SystemMetricsAggregatorConcrete::pushPowerStats(const PowerStats& stats) {
//     pushMetricsHelper<PowerStats>(stats,
//         [](SystemMetricsSnapshot& snap, const PowerStats& data) {
//             snap.powerStats = data;
//         });
// }



// // CSV Export Implementation
// // CSV Export Implementation
// void SystemMetricsAggregatorConcrete::exportToCSV(const std::string& filePath) {
//     std::lock_guard<std::mutex> lock(mutex_);
    
//     try {
//         std::ofstream file(filePath);
//         if(!file) throw std::runtime_error("Failed to open CSV file");

//         // Write CSV header
//         file << "Timestamp,"
//              << "CameraFrameNumber,CameraFPS,CameraWidth,CameraHeight,CameraFrameSize,"
//              << "AlgorithmInferenceTimeMs,AlgorithmConfidence,AlgorithmFPS,AlgorithmAvgProcTimeMs,AlgorithmTotalProcTime,"
//              << "DisplayLatencyMs,DroppedFrames,RenderTimeMs,"
//              << "RAM_Used_MB,RAM_Total_MB,SWAP_Used_MB,SWAP_Total_MB,Cached_MB,"
//              << "CPU1_Utilization_Percent,CPU1_Frequency_MHz,CPU2_Utilization_Percent,CPU2_Frequency_MHz,"
//              << "CPU3_Utilization_Percent,CPU3_Frequency_MHz,CPU4_Utilization_Percent,CPU4_Frequency_MHz,"
//              << "EMC_Frequency_Percent,GR3D_Frequency_Percent,"
//              << "PLL_Temp_C,CPU_Temp_C,PMIC_Temp_C,GPU_Temp_C,AO_Temp_C,Thermal_Temp_C,"
//              << "Power_Total_W,Power_Sensor0_W,Power_Sensor1_W,Power_Sensor2_W,Power_Sensor3_W\n";

//         // Write data rows
//         for(const auto& snap : snapshots_) {
//             file << formatTimestamp(snap.timestamp) << ","
//                  << snap.cameraStats.frameNumber << "," << snap.cameraStats.fps << ","
//                  << snap.cameraStats.frameWidth << "," << snap.cameraStats.frameHeight << "," 
//                  << snap.cameraStats.frameSize << ","
//                  << snap.algorithmStats.inferenceTimeMs << "," << snap.algorithmStats.confidenceScore << ","
//                  << snap.algorithmStats.fps_ << "," << snap.algorithmStats.avgProcTime_ << "," 
//                  << snap.algorithmStats.totalProcTime_ << ","
//                  << snap.displayStats.latencyMs << "," << snap.displayStats.droppedFrames << "," 
//                  << snap.displayStats.renderTimeMs << ","
//                  << snap.socInfo.RAM_In_Use_MB << "," << snap.socInfo.Total_RAM_MB << ","
//                  << snap.socInfo.SWAP_In_Use_MB << "," << snap.socInfo.Total_SWAP_MB << "," 
//                  << snap.socInfo.Cached_MB << ","
//                  << snap.socInfo.CPU1_Utilization_Percent << "," << snap.socInfo.CPU1_Frequency_MHz << ","
//                  << snap.socInfo.CPU2_Utilization_Percent << "," << snap.socInfo.CPU2_Frequency_MHz << ","
//                  << snap.socInfo.CPU3_Utilization_Percent << "," << snap.socInfo.CPU3_Frequency_MHz << ","
//                  << snap.socInfo.CPU4_Utilization_Percent << "," << snap.socInfo.CPU4_Frequency_MHz << ","
//                  << snap.socInfo.EMC_Frequency_Percent << "," << snap.socInfo.GR3D_Frequency_Percent << ","
//                  << snap.socInfo.PLL_Temperature_C << "," << snap.socInfo.CPU_Temperature_C << ","
//                  << snap.socInfo.PMIC_Temperature_C << "," << snap.socInfo.GPU_Temperature_C << ","
//                  << snap.socInfo.AO_Temperature_C << "," << snap.socInfo.Thermal_Temperature_C << ","
//                  << snap.powerStats.totalPower() << ","
//                  << snap.powerStats.sensorPower(0) << "," << snap.powerStats.sensorPower(1) << ","
//                  << snap.powerStats.sensorPower(2) << "," << snap.powerStats.sensorPower(3) << "\n";
//         }
//     } catch(const std::exception& e) {
//         spdlog::error("CSV export failed: {}", e.what());
//     }
// }


// // Export to JSON
// // This function exports the collected metrics to a JSON file

// // JSON Export Implementation
// void SystemMetricsAggregatorConcrete::exportToJSON(const std::string& filePath) {
//     std::lock_guard<std::mutex> lock(mutex_);
    
//     try {
//         json j;
//         for(const auto& snap : snapshots_) {
//             j.push_back(snapshotToJson(snap));
//         }

//         std::ofstream file(filePath);
//         if(!file) throw std::runtime_error("Failed to open JSON file");
//         file << j.dump(4);
//     } catch(const std::exception& e) {
//         spdlog::error("JSON export failed: {}", e.what());
//     }
// }

// // Real-time Append Functions
// void SystemMetricsAggregatorConcrete::appendSnapshotToCSV(
//     const SystemMetricsSnapshot& snap, 
//     const std::string& filePath) 
// {
//     std::lock_guard<std::mutex> lock(mutex_);
    
//     try {
//         bool writeHeader = false;
//         {
//             std::ifstream check(filePath);
//             writeHeader = check.peek() == std::ifstream::traits_type::eof();
//         }

//         std::ofstream file(filePath, std::ios::app);
//         if(!file) return;

//         if(writeHeader) {
//             file << "Timestamp,"
//                  << "CameraFrameNumber,CameraFPS,CameraWidth,CameraHeight,CameraFrameSize,"
//                  << "AlgorithmInferenceTimeMs,AlgorithmConfidence,AlgorithmFPS,AlgorithmAvgProcTimeMs,AlgorithmTotalProcTime,"
//                  << "DisplayLatencyMs,DroppedFrames,RenderTimeMs,"
//                  << "RAM_Used_MB,RAM_Total_MB,SWAP_Used_MB,SWAP_Total_MB,Cached_MB,"
//                  << "CPU1_Utilization_Percent,CPU1_Frequency_MHz,CPU2_Utilization_Percent,CPU2_Frequency_MHz,"
//                  << "CPU3_Utilization_Percent,CPU3_Frequency_MHz,CPU4_Utilization_Percent,CPU4_Frequency_MHz,"
//                  << "EMC_Frequency_Percent,GR3D_Frequency_Percent,"
//                  << "PLL_Temp_C,CPU_Temp_C,PMIC_Temp_C,GPU_Temp_C,AO_Temp_C,Thermal_Temp_C,"
//                  << "Power_Total_W,Power_Sensor0_W,Power_Sensor1_W,Power_Sensor2_W,Power_Sensor3_W\n";
//         }

//         file << formatTimestamp(snap.timestamp) << ","
//              << snap.cameraStats.frameNumber << "," << snap.cameraStats.fps << ","
//              << snap.cameraStats.frameWidth << "," << snap.cameraStats.frameHeight << "," 
//              << snap.cameraStats.frameSize << ","
//              << snap.algorithmStats.inferenceTimeMs << "," << snap.algorithmStats.confidenceScore << ","
//              << snap.algorithmStats.fps_ << "," << snap.algorithmStats.avgProcTime_ << "," 
//              << snap.algorithmStats.totalProcTime_ << ","
//              << snap.displayStats.latencyMs << "," << snap.displayStats.droppedFrames << "," 
//              << snap.displayStats.renderTimeMs << ","
//              << snap.socInfo.RAM_In_Use_MB << "," << snap.socInfo.Total_RAM_MB << ","
//              << snap.socInfo.SWAP_In_Use_MB << "," << snap.socInfo.Total_SWAP_MB << "," 
//              << snap.socInfo.Cached_MB << ","
//              << snap.socInfo.CPU1_Utilization_Percent << "," << snap.socInfo.CPU1_Frequency_MHz << ","
//              << snap.socInfo.CPU2_Utilization_Percent << "," << snap.socInfo.CPU2_Frequency_MHz << ","
//              << snap.socInfo.CPU3_Utilization_Percent << "," << snap.socInfo.CPU3_Frequency_MHz << ","
//              << snap.socInfo.CPU4_Utilization_Percent << "," << snap.socInfo.CPU4_Frequency_MHz << ","
//              << snap.socInfo.EMC_Frequency_Percent << "," << snap.socInfo.GR3D_Frequency_Percent << ","
//              << snap.socInfo.PLL_Temperature_C << "," << snap.socInfo.CPU_Temperature_C << ","
//              << snap.socInfo.PMIC_Temperature_C << "," << snap.socInfo.GPU_Temperature_C << ","
//              << snap.socInfo.AO_Temperature_C << "," << snap.socInfo.Thermal_Temperature_C << ","
//              << snap.powerStats.totalPower() << ","
//              << snap.powerStats.sensorPower(0) << "," << snap.powerStats.sensorPower(1) << ","
//              << snap.powerStats.sensorPower(2) << "," << snap.powerStats.sensorPower(3) << "\n";
//     } catch(const std::exception& e) {
//         spdlog::error("CSV append failed: {}", e.what());
//     }
// }

// void SystemMetricsAggregatorConcrete::appendSnapshotToJSON(
//     const SystemMetricsSnapshot& snap,
//     const std::string& filePath) 
// {
//     std::lock_guard<std::mutex> lock(mutex_);
    
//     try {
//         std::ofstream file(filePath, std::ios::app);
//         if(!file) return;
        
//         json j = snapshotToJson(snap);
//         file << j.dump() << "\n";  // NDJSON format
//     } catch(const std::exception& e) {
//         spdlog::error("JSON append failed: {}", e.what());
//     }
// }


// //
// void SystemMetricsAggregatorConcrete::pushMetrics(
//     const std::chrono::system_clock::time_point& timestamp,
//     std::function<void(SystemMetricsSnapshot&)> updateFn
// ) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     SystemMetricsSnapshot snapshot;
//     snapshot.timestamp = timestamp;
//     updateFn(snapshot);
//     snapshots_.push_back(snapshot);

//     // Append to both CSV and JSON
//     appendSnapshotToCSV(snapshot, "realtime_metrics.csv");
//     appendSnapshotToJSON(snapshot, "realtime_metrics.json");
// }

// void SystemMetricsAggregatorConcrete::appendSnapshotToCSV(
//     const SystemMetricsSnapshot& snap, 
//     const std::string& filePath
// ) {
//     std::ofstream file(filePath, std::ios::app);
//     if (!file.is_open()) return;

//     auto timeT = std::chrono::system_clock::to_time_t(snap.timestamp);
//     file << std::put_time(std::localtime(&timeT), "%Y-%m-%d %H:%M:%S") << ","
//         << snap.cameraStats.frameNumber << "," << snap.cameraStats.fps << ","
//         << snap.cameraStats.frameWidth << "," << snap.cameraStats.frameHeight << "," << snap.cameraStats.frameSize << ","
//         << snap.algorithmStats.inferenceTimeMs << "," << snap.algorithmStats.confidenceScore << ","
//         << snap.algorithmStats.fps_ << "," << snap.algorithmStats.avgProcTime_ << "," << snap.algorithmStats.totalProcTime_ << ","
//         << snap.displayStats.latencyMs << "," << snap.displayStats.droppedFrames << "," << snap.displayStats.renderTimeMs << ","
//         << snap.socInfo.RAM_In_Use_MB << "," << snap.socInfo.Total_RAM_MB << ","
//         << snap.socInfo.SWAP_In_Use_MB << "," << snap.socInfo.Total_SWAP_MB << "," << snap.socInfo.Cached_MB << ","
//         << snap.socInfo.CPU1_Utilization_Percent << "," << snap.socInfo.CPU1_Frequency_MHz << ","
//         << snap.socInfo.CPU2_Utilization_Percent << "," << snap.socInfo.CPU2_Frequency_MHz << ","
//         << snap.socInfo.CPU3_Utilization_Percent << "," << snap.socInfo.CPU3_Frequency_MHz << ","
//         << snap.socInfo.CPU4_Utilization_Percent << "," << snap.socInfo.CPU4_Frequency_MHz << ","
//         << snap.socInfo.EMC_Frequency_Percent << "," << snap.socInfo.GR3D_Frequency_Percent << ","
//         << snap.socInfo.PLL_Temperature_C << "," << snap.socInfo.CPU_Temperature_C << ","
//         << snap.socInfo.PMIC_Temperature_C << "," << snap.socInfo.GPU_Temperature_C << ","
//         << snap.socInfo.AO_Temperature_C << "," << snap.socInfo.Thermal_Temperature_C << ","
//         << snap.powerStats.totalPower() << ","
//         << snap.powerStats.sensorPower(0) << "," << snap.powerStats.sensorPower(1) << ","
//         << snap.powerStats.sensorPower(2) << "," << snap.powerStats.sensorPower(3) << ","
//         << snap.powerStats.voltages.size() << "," << snap.powerStats.currents.size()
//         << "\n";
// }

// void SystemMetricsAggregatorConcrete::appendSnapshotToJSON(
//     const SystemMetricsSnapshot& snap,
//     const std::string& filePath
// ) {
//     json snapshotJson;
//     // ... (Same JSON construction as in exportToJSON, but for single snapshot)
    
//     std::ofstream file(filePath, std::ios::app);
//     if (file.tellp() == 0) {
//         file << "[\n";
//     } else {
//         file.seekp(-2, std::ios::end);
//         file << ",\n";
//     }
//     file << snapshotJson.dump(4) << "\n]";
// }

//====================================================================
