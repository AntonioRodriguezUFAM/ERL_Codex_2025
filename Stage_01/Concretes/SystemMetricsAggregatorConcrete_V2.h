// SystemMetricsAggregatorConcrete_v2.h
// ------------------------------------------------------------------
// Drop?in replacement for SystemMetricsAggregatorConcrete that provides
// * frame?centric aggregation (one CSV row per frame)
// * derived timing metrics (processingLatency, displayLatency, endToEndLatency)
// * per?frame energy estimation (joulesPerFrame)
//
// Integration notes
// -----------------
// ? Camera thread calls  beginFrame(frameId,cameraStats)
// ? Algorithm thread calls mergeAlgorithm(frameId,algorithmStats)
// ? Display thread  calls mergeDisplay (frameId,displayStats) ? this flushes
//   the fully?populated snapshot to CSV/JSON using the latest SoC & power.
// ? Power & SoC threads continue to push their data independently.  The
//   aggregator keeps the *most recent* sample and attaches it to every frame
//   at flush time.
// ? All public methods are thread?safe.
// ------------------------------------------------------------------
#pragma once

#include <unordered_map>
#include <fstream>
#include <mutex>
#include <iomanip>
#include <sstream>
#include "../Interfaces/ISystemMetricsAggregator.h"
#include "../SharedStructures/allModulesStatcs.h"
//#include <nlohmann/json.hpp>
#include "../nlohmann/json.hpp"
#include <spdlog/spdlog.h>

//SampleTestIntegrationCode_v2.cpp
using json = nlohmann::json;

/**
 * @class SystemMetricsAggregatorConcreteV2
 * @brief Frame-centric metrics aggregator with thread-safe operations and CSV/JSON export.
 * Camera thread calls beginFrame, algorithm thread calls mergeAlgorithm, and display thread
 * calls mergeDisplay to flush a complete snapshot. SoC and power threads push data
 * independently, with the latest values attached to each frame.
 */
class SystemMetricsAggregatorConcreteV2 : public ISystemMetricsAggregator {
public:
    // Constructor
    SystemMetricsAggregatorConcreteV2()
        : csvPath_("realtime_metrics.csv"), jsonPath_("realtime_metrics.ndjson"),
          retentionWindow_(std::chrono::seconds(60)) {}

    // Frame-centric API
    void beginFrame(uint64_t frameId, const CameraStats& stats) override;
    void mergeAlgorithm(uint64_t frameId, const AlgorithmStats& stats) override;
    void mergeDisplay(uint64_t frameId, const DisplayStats& stats) override;

    // SoC and power updates
    void pushPowerStats(const PowerStats& stats) override;
    void pushSoCStats(const JetsonNanoInfo& stats) override;

    // Generic metrics push
    void pushMetrics(
        const std::chrono::system_clock::time_point& timestamp,
        std::function<void(SystemMetricsSnapshot&)> updateFn) override;

    // Legacy interface stubs
    void pushCameraStats(const CameraStats&) override {}
    void pushAlgorithmStats(const AlgorithmStats&) override {}
    void pushDisplayStats(const DisplayStats&) override {}

    // Snapshot accessors
    SystemMetricsSnapshot getLatestSnapshot() const override;
    std::vector<SystemMetricsSnapshot> getAllSnapshots() const override;
    SystemMetricsSnapshot getAggregatedAt(std::chrono::system_clock::time_point) const override {
        return {};
    }

    // Export methods
    void exportToCSV(const std::string& filePath) override;
    void exportToJSON(const std::string& filePath) override;

    // Configuration
    void setCsvPath(const std::string& p) { csvPath_ = p; }
    void setJsonPath(const std::string& p) { jsonPath_ = p; }
    void setRetentionWindow(std::chrono::seconds window) { retentionWindow_ = window; }

private:
    struct PartialFrame {
        CameraStats cam;
        bool hasCam = false;
        AlgorithmStats alg;
        bool hasAlg = false;
        DisplayStats disp;
        bool hasDisp = false;
    };

    template<typename T>
    void pushMetricsHelper(
        const T& data,
        std::function<void(SystemMetricsSnapshot&, const T&)> updateFn);

    void finalizeFrame(uint64_t frameId, PartialFrame&& pf);
    void appendSnapshotToCSV(const SystemMetricsSnapshot& snap);
    void appendSnapshotToJSON(const SystemMetricsSnapshot& snap);
    std::string formatTimestamp(const std::chrono::system_clock::time_point& tp) const;
    json snapshotToJson(const SystemMetricsSnapshot& snap) const;
    void enforceRetentionPolicy();

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, PartialFrame> pending_;
    std::vector<SystemMetricsSnapshot> history_;
    PowerStats latestPower_;
    JetsonNanoInfo latestSoC_;
    std::string csvPath_;
    std::string jsonPath_;
    std::chrono::seconds retentionWindow_;
};

// Implementation
inline void SystemMetricsAggregatorConcreteV2::pushMetrics(
    const std::chrono::system_clock::time_point& timestamp,
    std::function<void(SystemMetricsSnapshot&)> updateFn) {
    SystemMetricsSnapshot snapshot(timestamp);
    updateFn(snapshot);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.push_back(snapshot);
        enforceRetentionPolicy();
    }
    appendSnapshotToCSV(snapshot);
    appendSnapshotToJSON(snapshot);
}

template<typename T>
inline void SystemMetricsAggregatorConcreteV2::pushMetricsHelper(
    const T& data,
    std::function<void(SystemMetricsSnapshot&, const T&)> updateFn) {
    pushMetrics(data.timestamp, [&](auto& snap) { updateFn(snap, data); });
}

inline void SystemMetricsAggregatorConcreteV2::enforceRetentionPolicy() {
    if (retentionWindow_ <= std::chrono::seconds(0)) return;
    auto cutoff = std::chrono::system_clock::now() - retentionWindow_;
    std::lock_guard<std::mutex> lock(mutex_);
    history_.erase(
        std::remove_if(history_.begin(), history_.end(),
                       [&cutoff](const auto& snap) { return snap.timestamp < cutoff; }),
        history_.end());
}

inline void SystemMetricsAggregatorConcreteV2::exportToCSV(const std::string& filePath) {
    std::vector<SystemMetricsSnapshot> copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        copy = history_;
    }
    try {
        std::ofstream file(filePath);
        if (!file) throw std::runtime_error("Failed to open CSV file: " + filePath);

        file << "Timestamp,FrameNumber,FPS,FrameWidth,FrameHeight,FrameSize,"
             << "InferenceTimeMs,ConfidenceScore,AlgorithmFPS,AvgProcTimeMs,TotalProcTimeMs,GPUFreeMemoryMB,GPUTotalMemoryMB,CudaKernelTimeMs,"
             << "DisplayLatencyMs,DroppedFrames,RenderTimeMs,EndToEndLatencyMs,JoulesPerFrame,"
             << "RAM_In_Use_MB,Total_RAM_MB,LFB_Size_MB,Block_Max_MB,SWAP_In_Use_MB,Total_SWAP_MB,Cached_MB,"
             << "used_IRAM_kB,total_IRAM_kB,lfb_kB,"
             << "CPU1_Utilization_Percent,CPU1_Frequency_MHz,CPU2_Utilization_Percent,CPU2_Frequency_MHz,"
             << "CPU3_Utilization_Percent,CPU3_Frequency_MHz,CPU4_Utilization_Percent,CPU4_Frequency_MHz,"
             << "EMC_Frequency_Percent,GR3D_Frequency_Percent,"
             << "PLL_Temperature_C,CPU_Temperature_C,PMIC_Temperature_C,GPU_Temperature_C,AO_Temperature_C,Thermal_Temperature_C,"
             << "PowerTotalW,PowerAverageW,PowerSensor0W,PowerSensor0V,PowerSensor0A,PowerSensor1W,PowerSensor1V,PowerSensor1A,"
             << "PowerSensor2W,PowerSensor2V,PowerSensor2A,PowerSensor3W,PowerSensor3V,PowerSensor3A\n";

        for (const auto& snap : copy) {
            file << formatTimestamp(snap.timestamp) << ","
                 << snap.cameraStats.frameNumber << "," << snap.cameraStats.fps << ","
                 << snap.cameraStats.frameWidth << "," << snap.cameraStats.frameHeight << ","
                 << snap.cameraStats.frameSize << ","
                 << snap.algorithmStats.inferenceTimeMs << "," << snap.algorithmStats.confidenceScore << ","
                 << snap.algorithmStats.fps << "," << snap.algorithmStats.avgProcTimeMs << ","
                 << snap.algorithmStats.totalProcTimeMs << ","
                 << snap.algorithmStats.gpuFreeMemory / (1024.0 * 1024.0) << ","
                 << snap.algorithmStats.gpuTotalMemory / (1024.0 * 1024.0) << ","
                 << snap.algorithmStats.cudaKernelTimeMs << ","
                 << snap.displayStats.latencyMs << "," << snap.displayStats.droppedFrames << ","
                 << snap.displayStats.renderTimeMs << "," << snap.endToEndLatencyMs << ","
                 << snap.joulesPerFrame << ","
                 << snap.socInfo.RAM_In_Use_MB << "," << snap.socInfo.Total_RAM_MB << ","
                 << snap.socInfo.LFB_Size_MB << "," << snap.socInfo.Block_Max_MB << ","
                 << snap.socInfo.SWAP_In_Use_MB << "," << snap.socInfo.Total_SWAP_MB << ","
                 << snap.socInfo.Cached_MB << ","
                 << snap.socInfo.used_IRAM_kB << "," << snap.socInfo.total_IRAM_kB << "," << snap.socInfo.lfb_kB << ","
                 << snap.socInfo.CPU1_Utilization_Percent << "," << snap.socInfo.CPU1_Frequency_MHz << ","
                 << snap.socInfo.CPU2_Utilization_Percent << "," << snap.socInfo.CPU2_Frequency_MHz << ","
                 << snap.socInfo.CPU3_Utilization_Percent << "," << snap.socInfo.CPU3_Frequency_MHz << ","
                 << snap.socInfo.CPU4_Utilization_Percent << "," << snap.socInfo.CPU4_Frequency_MHz << ","
                 << snap.socInfo.EMC_Frequency_Percent << "," << snap.socInfo.GR3D_Frequency_Percent << ","
                 << snap.socInfo.PLL_Temperature_C << "," << snap.socInfo.CPU_Temperature_C << ","
                 << snap.socInfo.PMIC_Temperature_C << "," << snap.socInfo.GPU_Temperature_C << ","
                 << snap.socInfo.AO_Temperature_C << "," << snap.socInfo.Thermal_Temperature_C << ","
                 << snap.powerStats.totalPower() << "," << snap.powerStats.averagePower() << ","
                 << snap.powerStats.sensorPower(0) << "," << (snap.powerStats.sensorCount() > 0 ? snap.powerStats.voltages[0] : 0.0) << ","
                 << (snap.powerStats.sensorCount() > 0 ? snap.powerStats.currents[0] : 0.0) << ","
                 << snap.powerStats.sensorPower(1) << "," << (snap.powerStats.sensorCount() > 1 ? snap.powerStats.voltages[1] : 0.0) << ","
                 << (snap.powerStats.sensorCount() > 1 ? snap.powerStats.currents[1] : 0.0) << ","
                 << snap.powerStats.sensorPower(2) << "," << (snap.powerStats.sensorCount() > 2 ? snap.powerStats.voltages[2] : 0.0) << ","
                 << (snap.powerStats.sensorCount() > 2 ? snap.powerStats.currents[2] : 0.0) << ","
                 << snap.powerStats.sensorPower(3) << "," << (snap.powerStats.sensorCount() > 3 ? snap.powerStats.voltages[3] : 0.0) << ","
                 << (snap.powerStats.sensorCount() > 3 ? snap.powerStats.currents[3] : 0.0) << "\n";
        }
    } catch (const std::exception& e) {
        spdlog::error("CSV export failed: {}", e.what());
        throw;
    }
}

inline void SystemMetricsAggregatorConcreteV2::exportToJSON(const std::string& filePath) {
    std::vector<SystemMetricsSnapshot> copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        copy = history_;
    }
    try {
        json j = json::array();
        for (const auto& snap : copy) {
            j.push_back(snapshotToJson(snap));
        }
        std::ofstream file(filePath);
        if (!file) throw std::runtime_error("Failed to open JSON file: " + filePath);
        file << j.dump(4);
    } catch (const std::exception& e) {
        spdlog::error("JSON export failed: {}", e.what());
        throw;
    }
}

inline void SystemMetricsAggregatorConcreteV2::beginFrame(uint64_t id, const CameraStats& s) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& pf = pending_[id];
    pf.cam = s;
    pf.hasCam = true;
}

inline void SystemMetricsAggregatorConcreteV2::mergeAlgorithm(uint64_t id, const AlgorithmStats& s) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pending_.find(id);
    if (it == pending_.end()) {
        spdlog::warn("[Aggregator] mergeAlgorithm: frame {} not begun", id);
        return;
    }
    it->second.alg = s;
    it->second.hasAlg = true;
}

inline void SystemMetricsAggregatorConcreteV2::mergeDisplay(uint64_t id, const DisplayStats& s) {
    PartialFrame pf;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pending_.find(id);
        if (it == pending_.end()) {
            spdlog::warn("[Aggregator] mergeDisplay: frame {} not begun", id);
            return;
        }
        pf = std::move(it->second);
        pf.disp = s;
        pf.hasDisp = true;
        pending_.erase(it);
    }
    finalizeFrame(id, std::move(pf));
}

inline void SystemMetricsAggregatorConcreteV2::pushPowerStats(const PowerStats& s) {
    std::lock_guard<std::mutex> lock(mutex_);
    latestPower_ = s;
}

inline void SystemMetricsAggregatorConcreteV2::pushSoCStats(const JetsonNanoInfo& s) {
    std::lock_guard<std::mutex> lock(mutex_);
    latestSoC_ = s;
}

inline void SystemMetricsAggregatorConcreteV2::finalizeFrame(uint64_t id, PartialFrame&& pf) {
    if (!(pf.hasCam && pf.hasAlg && pf.hasDisp)) {
        spdlog::warn("[Aggregator] finalizeFrame({}): incomplete data (cam:{} alg:{} disp:{})",
                     id, pf.hasCam, pf.hasAlg, pf.hasDisp);
        return;
    }

    SystemMetricsSnapshot snap(pf.cam.timestamp);
    snap.cameraStats = pf.cam;
    snap.algorithmStats = pf.alg;
    snap.displayStats = pf.disp;
    snap.socInfo = latestSoC_;
    snap.powerStats = latestPower_;

    // Derived timing metrics
   // double processingLatency = pf.alg.inferenceTimeMs;
    //double displayLatency = pf.disp.renderTimeMs;
    auto dtEndToEnd = std::chrono::duration<double, std::milli>(pf.disp.timestamp - pf.cam.timestamp).count();
    snap.endToEndLatencyMs = dtEndToEnd;

    // Energy per frame
    double framePeriod = (pf.cam.fps > 0.0) ? (1.0 / pf.cam.fps) : 0.0; // seconds
    snap.joulesPerFrame = latestPower_.totalPower() * framePeriod;

    // Commit
    {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.push_back(snap);
        enforceRetentionPolicy();
    }
    appendSnapshotToCSV(snap);
    appendSnapshotToJSON(snap);
}

inline SystemMetricsSnapshot SystemMetricsAggregatorConcreteV2::getLatestSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return history_.empty() ? SystemMetricsSnapshot(std::chrono::system_clock::now()) : history_.back();
}

inline std::vector<SystemMetricsSnapshot> SystemMetricsAggregatorConcreteV2::getAllSnapshots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return history_;
}

inline void SystemMetricsAggregatorConcreteV2::appendSnapshotToCSV(const SystemMetricsSnapshot& snap) {
    std::ofstream file(csvPath_, std::ios::app);
    if (!file) {
        spdlog::error("[Aggregator] Cannot open {}", csvPath_);
        return;
    }

    static bool headerWritten = false;
    if (!headerWritten) {
        file << "Timestamp,FrameNumber,FPS,FrameWidth,FrameHeight,FrameSize,"
             << "InferenceTimeMs,ConfidenceScore,AlgorithmFPS,AvgProcTimeMs,TotalProcTimeMs,GPUFreeMemoryMB,GPUTotalMemoryMB,CudaKernelTimeMs,"
             << "DisplayLatencyMs,DroppedFrames,RenderTimeMs,EndToEndLatencyMs,JoulesPerFrame,"
             << "RAM_In_Use_MB,Total_RAM_MB,LFB_Size_MB,Block_Max_MB,SWAP_In_Use_MB,Total_SWAP_MB,Cached_MB,"
             << "used_IRAM_kB,total_IRAM_kB,lfb_kB,"
             << "CPU1_Utilization_Percent,CPU1_Frequency_MHz,CPU2_Utilization_Percent,CPU2_Frequency_MHz,"
             << "CPU3_Utilization_Percent,CPU3_Frequency_MHz,CPU4_Utilization_Percent,CPU4_Frequency_MHz,"
             << "EMC_Frequency_Percent,GR3D_Frequency_Percent,"
             << "PLL_Temperature_C,CPU_Temperature_C,PMIC_Temperature_C,GPU_Temperature_C,AO_Temperature_C,Thermal_Temperature_C,"
             << "PowerTotalW,PowerAverageW,PowerSensor0W,PowerSensor0V,PowerSensor0A,PowerSensor1W,PowerSensor1V,PowerSensor1A,"
             << "PowerSensor2W,PowerSensor2V,PowerSensor2A,PowerSensor3W,PowerSensor3V,PowerSensor3A\n";
        headerWritten = true;
    }

    file << formatTimestamp(snap.timestamp) << ","
         << snap.cameraStats.frameNumber << "," << snap.cameraStats.fps << ","
         << snap.cameraStats.frameWidth << "," << snap.cameraStats.frameHeight << ","
         << snap.cameraStats.frameSize << ","
         << snap.algorithmStats.inferenceTimeMs << "," << snap.algorithmStats.confidenceScore << ","
         << snap.algorithmStats.fps << "," << snap.algorithmStats.avgProcTimeMs << ","
         << snap.algorithmStats.totalProcTimeMs << ","
         << snap.algorithmStats.gpuFreeMemory / (1024.0 * 1024.0) << ","
         << snap.algorithmStats.gpuTotalMemory / (1024.0 * 1024.0) << ","
         << snap.algorithmStats.cudaKernelTimeMs << ","
         << snap.displayStats.latencyMs << "," << snap.displayStats.droppedFrames << ","
         << snap.displayStats.renderTimeMs << "," << snap.endToEndLatencyMs << ","
         << snap.joulesPerFrame << ","
         << snap.socInfo.RAM_In_Use_MB << "," << snap.socInfo.Total_RAM_MB << ","
         << snap.socInfo.LFB_Size_MB << "," << snap.socInfo.Block_Max_MB << ","
         << snap.socInfo.SWAP_In_Use_MB << "," << snap.socInfo.Total_SWAP_MB << ","
         << snap.socInfo.Cached_MB << ","
         << snap.socInfo.used_IRAM_kB << "," << snap.socInfo.total_IRAM_kB << "," << snap.socInfo.lfb_kB << ","
         << snap.socInfo.CPU1_Utilization_Percent << "," << snap.socInfo.CPU1_Frequency_MHz << ","
         << snap.socInfo.CPU2_Utilization_Percent << "," << snap.socInfo.CPU2_Frequency_MHz << ","
         << snap.socInfo.CPU3_Utilization_Percent << "," << snap.socInfo.CPU3_Frequency_MHz << ","
         << snap.socInfo.CPU4_Utilization_Percent << "," << snap.socInfo.CPU4_Frequency_MHz << ","
         << snap.socInfo.EMC_Frequency_Percent << "," << snap.socInfo.GR3D_Frequency_Percent << ","
         << snap.socInfo.PLL_Temperature_C << "," << snap.socInfo.CPU_Temperature_C << ","
         << snap.socInfo.PMIC_Temperature_C << "," << snap.socInfo.GPU_Temperature_C << ","
         << snap.socInfo.AO_Temperature_C << "," << snap.socInfo.Thermal_Temperature_C << ","
         << snap.powerStats.totalPower() << "," << snap.powerStats.averagePower() << ","
         << snap.powerStats.sensorPower(0) << "," << (snap.powerStats.sensorCount() > 0 ? snap.powerStats.voltages[0] : 0.0) << ","
         << (snap.powerStats.sensorCount() > 0 ? snap.powerStats.currents[0] : 0.0) << ","
         << snap.powerStats.sensorPower(1) << "," << (snap.powerStats.sensorCount() > 1 ? snap.powerStats.voltages[1] : 0.0) << ","
         << (snap.powerStats.sensorCount() > 1 ? snap.powerStats.currents[1] : 0.0) << ","
         << snap.powerStats.sensorPower(2) << "," << (snap.powerStats.sensorCount() > 2 ? snap.powerStats.voltages[2] : 0.0) << ","
         << (snap.powerStats.sensorCount() > 2 ? snap.powerStats.currents[2] : 0.0) << ","
         << snap.powerStats.sensorPower(3) << "," << (snap.powerStats.sensorCount() > 3 ? snap.powerStats.voltages[3] : 0.0) << ","
         << (snap.powerStats.sensorCount() > 3 ? snap.powerStats.currents[3] : 0.0) << "\n";
}

inline void SystemMetricsAggregatorConcreteV2::appendSnapshotToJSON(const SystemMetricsSnapshot& snap) {
    std::ofstream file(jsonPath_, std::ios::app);
    if (!file) {
        spdlog::error("[Aggregator] Cannot open {}", jsonPath_);
        return;
    }
    file << snapshotToJson(snap).dump() << "\n";
}

inline std::string SystemMetricsAggregatorConcreteV2::formatTimestamp(
    const std::chrono::system_clock::time_point& tp) const {
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

inline json SystemMetricsAggregatorConcreteV2::snapshotToJson(const SystemMetricsSnapshot& snap) const {
    json j;
    j["timestamp"] = formatTimestamp(snap.timestamp);
    j["camera"] = {
        {"frame_number", snap.cameraStats.frameNumber},
        {"fps", snap.cameraStats.fps},
        {"width", snap.cameraStats.frameWidth},
        {"height", snap.cameraStats.frameHeight},
        {"frame_size", snap.cameraStats.frameSize}
    };
    j["algorithm"] = {
        {"inference_time_ms", snap.algorithmStats.inferenceTimeMs},
        {"confidence", snap.algorithmStats.confidenceScore},
        {"fps", snap.algorithmStats.fps},
        {"avg_proc_time_ms", snap.algorithmStats.avgProcTimeMs},
        {"total_proc_time_ms", snap.algorithmStats.totalProcTimeMs},
        {"gpu_free_memory_mb", snap.algorithmStats.gpuFreeMemory / (1024.0 * 1024.0)},
        {"gpu_total_memory_mb", snap.algorithmStats.gpuTotalMemory / (1024.0 * 1024.0)},
        {"cuda_kernel_time_ms", snap.algorithmStats.cudaKernelTimeMs}
    };
    j["display"] = {
        {"latency_ms", snap.displayStats.latencyMs},
        {"dropped_frames", snap.displayStats.droppedFrames},
        {"render_time_ms", snap.displayStats.renderTimeMs}
    };
    j["derived"] = {
        {"end_to_end_latency_ms", snap.endToEndLatencyMs},
        {"joules_per_frame", snap.joulesPerFrame}
    };
    j["soc"] = {
        {"ram_in_use_mb", snap.socInfo.RAM_In_Use_MB},
        {"total_ram_mb", snap.socInfo.Total_RAM_MB},
        {"lfb_size_mb", snap.socInfo.LFB_Size_MB},
        {"block_max_mb", snap.socInfo.Block_Max_MB},
        {"swap_in_use_mb", snap.socInfo.SWAP_In_Use_MB},
        {"total_swap_mb", snap.socInfo.Total_SWAP_MB},
        {"cached_mb", snap.socInfo.Cached_MB},
        {"used_iram_kb", snap.socInfo.used_IRAM_kB},
        {"total_iram_kb", snap.socInfo.total_IRAM_kB},
        {"lfb_kb", snap.socInfo.lfb_kB},
        {"cpu1_utilization_percent", snap.socInfo.CPU1_Utilization_Percent},
        {"cpu1_frequency_mhz", snap.socInfo.CPU1_Frequency_MHz},
        {"cpu2_utilization_percent", snap.socInfo.CPU2_Utilization_Percent},
        {"cpu2_frequency_mhz", snap.socInfo.CPU2_Frequency_MHz},
        {"cpu3_utilization_percent", snap.socInfo.CPU3_Utilization_Percent},
        {"cpu3_frequency_mhz", snap.socInfo.CPU3_Frequency_MHz},
        {"cpu4_utilization_percent", snap.socInfo.CPU4_Utilization_Percent},
        {"cpu4_frequency_mhz", snap.socInfo.CPU4_Frequency_MHz},
        {"emc_frequency_percent", snap.socInfo.EMC_Frequency_Percent},
        {"gr3d_frequency_percent", snap.socInfo.GR3D_Frequency_Percent},
        {"pll_temperature_c", snap.socInfo.PLL_Temperature_C},
        {"cpu_temperature_c", snap.socInfo.CPU_Temperature_C},
        {"pmic_temperature_c", snap.socInfo.PMIC_Temperature_C},
        {"gpu_temperature_c", snap.socInfo.GPU_Temperature_C},
        {"ao_temperature_c", snap.socInfo.AO_Temperature_C},
        {"thermal_temperature_c", snap.socInfo.Thermal_Temperature_C}
    };
    j["power"] = {
        {"total_w", snap.powerStats.totalPower()},
        {"average_w", snap.powerStats.averagePower()},
        {"sensor0", {
            {"power_w", snap.powerStats.sensorPower(0)},
            {"voltage_v", snap.powerStats.sensorCount() > 0 ? snap.powerStats.voltages[0] : 0.0},
            {"current_a", snap.powerStats.sensorCount() > 0 ? snap.powerStats.currents[0] : 0.0}
        }},
        {"sensor1", {
            {"power_w", snap.powerStats.sensorPower(1)},
            {"voltage_v", snap.powerStats.sensorCount() > 1 ? snap.powerStats.voltages[1] : 0.0},
            {"current_a", snap.powerStats.sensorCount() > 1 ? snap.powerStats.currents[1] : 0.0}
        }},
        {"sensor2", {
            {"power_w", snap.powerStats.sensorPower(2)},
            {"voltage_v", snap.powerStats.sensorCount() > 2 ? snap.powerStats.voltages[2] : 0.0},
            {"current_a", snap.powerStats.sensorCount() > 2 ? snap.powerStats.currents[2] : 0.0}
        }},
        {"sensor3", {
            {"power_w", snap.powerStats.sensorPower(3)},
            {"voltage_v", snap.powerStats.sensorCount() > 3 ? snap.powerStats.voltages[3] : 0.0},
            {"current_a", snap.powerStats.sensorCount() > 3 ? snap.powerStats.currents[3] : 0.0}
        }}
    };
    return j;
}

//#endif // SYSTEM_METRICS_AGGREGATOR_CONCRETE_V2_H