#pragma once

#include <chrono>
#include <vector>
#include <cstdint>
#include <string>
#include "../Includes/lynsyn.h" // Lynsyn library for power measurements

/**
 * @struct PowerStats
 * @brief Represents power measurements across multiple sensors at a single point in time.
 */
struct PowerStats {
    std::chrono::system_clock::time_point timestamp;
    std::vector<double> voltages; // Volts, indexed by sensor ID (0-3)
    std::vector<double> currents; // Amps, indexed by sensor ID (0-3)

    /**
     * @brief Power for a specific sensor (P = V * I).
     * @param i Sensor index.
     * @return Power in watts.
     */
    double sensorPower(size_t i) const {
        if (i < voltages.size() && i < currents.size()) {
            return voltages[i] * currents[i];
        }
        return 0.0;
    }

    /**
     * @brief Calculates power for each sensor (P = V * I).
     * @return Vector of power values in watts.
     */
    std::vector<double> powerPerSensor() const {
        std::vector<double> powers;
        size_t count = std::min(voltages.size(), currents.size());
        powers.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            powers.push_back(voltages[i] * currents[i]);
        }
        return powers;
    }

    /**
     * @brief Total power across all sensors.
     * @return Total power in watts.
     */
    double totalPower() const {
        double sum = 0.0;
        size_t count = std::min(voltages.size(), currents.size());
        for (size_t i = 0; i < count; ++i) {
            sum += voltages[i] * currents[i];
        }
        return sum;
    }

    /**
     * @brief Average power across all sensors.
     * @return Average power in watts.
     */
    double averagePower() const {
        size_t count = std::min(voltages.size(), currents.size());
        return count > 0 ? totalPower() / count : 0.0;
    }

    /**
     * @brief Number of active sensors.
     * @return Number of sensors with valid data.
     */
    size_t sensorCount() const {
        return std::min(voltages.size(), currents.size());
    }
};

/**
 * @brief Convert raw LynsynSample to PowerStats structure.
 * @param s LynsynSample containing voltage, current, and time data.
 * @return PowerStats with populated fields.
 */
inline PowerStats convertToPowerStats(const LynsynSample& s) {
    PowerStats stats;

    // Convert LynsynSample time (cycles) to timestamp
    double seconds = lynsyn_cyclesToSeconds(s.time);
    auto duration = std::chrono::duration<double>(seconds);
    stats.timestamp = std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(duration));

    // Populate voltages and currents, up to LYNSYN_MAX_SENSORS or 4 (per CSV)
    constexpr size_t max_sensors = std::min(static_cast<size_t>(LYNSYN_MAX_SENSORS), static_cast<size_t>(4));
    stats.voltages.reserve(max_sensors);
    stats.currents.reserve(max_sensors);

    for (size_t i = 0; i < max_sensors; ++i) {
        stats.voltages.push_back(s.voltage[i]);
        stats.currents.push_back(s.current[i]);
    }

    return stats;
}

/**
 * @struct CameraStats
 * @brief Metrics for camera performance.
 */
struct CameraStats {
    std::chrono::system_clock::time_point timestamp;
    uint64_t frameNumber = 0; // Frame sequence number
    double fps = 0.0;         // Frames per second
    uint32_t frameWidth = 0;  // Frame width in pixels
    uint32_t frameHeight = 0; // Frame height in pixels
    uint64_t frameSize = 0;   // Frame size in bytes

    CameraStats() = default;
    CameraStats(std::chrono::system_clock::time_point ts)
        : timestamp(ts), frameNumber(0), fps(0.0), frameWidth(0), frameHeight(0), frameSize(0) {}
    CameraStats(std::chrono::system_clock::time_point ts, uint64_t fn, double f, uint32_t w, uint32_t h, uint64_t s)
        : timestamp(ts), frameNumber(fn), fps(f), frameWidth(w), frameHeight(h), frameSize(s) {}
};

/**
 * @struct AlgorithmStats
 * @brief Metrics for algorithm processing (e.g., inference).
 */
struct AlgorithmStats {
    std::chrono::system_clock::time_point timestamp;
    double inferenceTimeMs = 0.0;     // Inference time per frame (ms)
    double confidenceScore = 0.0;     // Algorithm confidence (0-1)
    double fps = 0.0;                 // Algorithm processing FPS
    double avgProcTimeMs = 0.0;       // Average processing time per frame (ms)
    double totalProcTimeMs = 0.0;     // Total processing time (ms)
    uint64_t framesCount = 0;         // Number of processed frames
    uint64_t gpuFreeMemory = 0;       // GPU free memory (bytes)
    uint64_t gpuTotalMemory = 0;      // GPU total memory (bytes)
    double cudaKernelTimeMs = 0.0;    // CUDA kernel execution time (ms)

    AlgorithmStats() = default;
    AlgorithmStats(std::chrono::system_clock::time_point ts)
        : timestamp(ts), inferenceTimeMs(0.0), confidenceScore(0.0), fps(0.0), avgProcTimeMs(0.0),
          totalProcTimeMs(0.0), framesCount(0), gpuFreeMemory(0), gpuTotalMemory(0), cudaKernelTimeMs(0.0) {}
    AlgorithmStats(std::chrono::system_clock::time_point ts, double inf, double conf, double f, double avg, double total, uint64_t fc)
        : timestamp(ts), inferenceTimeMs(inf), confidenceScore(conf), fps(f), avgProcTimeMs(avg), totalProcTimeMs(total), framesCount(fc),
          gpuFreeMemory(0), gpuTotalMemory(0), cudaKernelTimeMs(0.0) {}
};

/**
 * @struct DisplayStats
 * @brief Metrics for display rendering.
 */
struct DisplayStats {
    std::chrono::system_clock::time_point timestamp;
    double latencyMs = 0.0;      // Display latency (ms)
    uint32_t droppedFrames = 0;  // Number of dropped frames
    double renderTimeMs = 0.0;   // Rendering time per frame (ms)

    DisplayStats() = default;
    DisplayStats(std::chrono::system_clock::time_point ts)
        : timestamp(ts), latencyMs(0.0), droppedFrames(0), renderTimeMs(0.0) {}
    DisplayStats(std::chrono::system_clock::time_point ts, double lat, uint32_t df, double rt)
        : timestamp(ts), latencyMs(lat), droppedFrames(df), renderTimeMs(rt) {}
};

/**
 * @struct JetsonNanoInfo
 * @brief System-on-Chip metrics for Jetson Nano.
 */
struct JetsonNanoInfo {
    std::chrono::system_clock::time_point timestamp;
    // Memory metrics (MB)
    double RAM_In_Use_MB = 0.0;      // RAM currently in use
    double Total_RAM_MB = 0.0;       // Total available RAM
    double LFB_Size_MB = 0.0;        // Largest free block size
    double Block_Max_MB = 0.0;       // Maximum block size
    double SWAP_In_Use_MB = 0.0;     // SWAP currently in use
    double Total_SWAP_MB = 0.0;      // Total available SWAP
    double Cached_MB = 0.0;          // Cached memory
    // IRAM metrics (kB)
    double used_IRAM_kB = 0.0;       // Internal RAM used
    double total_IRAM_kB = 0.0;      // Total internal RAM
    double lfb_kB = 0.0;             // Largest free block
    // CPU metrics (4 cores)
    double CPU1_Utilization_Percent = 0.0;
    double CPU1_Frequency_MHz = 0.0;
    double CPU2_Utilization_Percent = 0.0;
    double CPU2_Frequency_MHz = 0.0;
    double CPU3_Utilization_Percent = 0.0;
    double CPU3_Frequency_MHz = 0.0;
    double CPU4_Utilization_Percent = 0.0;
    double CPU4_Frequency_MHz = 0.0;
    // Memory and GPU frequency
    double EMC_Frequency_Percent = 0.0; // External Memory Controller
    double GR3D_Frequency_Percent = 0.0; // GPU frequency
    // Temperature sensors (°C)
    double PLL_Temperature_C = 0.0;
    double CPU_Temperature_C = 0.0;
    double PMIC_Temperature_C = 0.0;
    double GPU_Temperature_C = 0.0;
    double AO_Temperature_C = 0.0; // Always-On sensor
    double Thermal_Temperature_C = 0.0;

    JetsonNanoInfo() = default;
    JetsonNanoInfo(std::chrono::system_clock::time_point ts)
        : timestamp(ts),
          RAM_In_Use_MB(0.0), Total_RAM_MB(0.0), LFB_Size_MB(0.0), Block_Max_MB(0.0),
          SWAP_In_Use_MB(0.0), Total_SWAP_MB(0.0), Cached_MB(0.0),
          used_IRAM_kB(0.0), total_IRAM_kB(0.0), lfb_kB(0.0),
          CPU1_Utilization_Percent(0.0), CPU1_Frequency_MHz(0.0),
          CPU2_Utilization_Percent(0.0), CPU2_Frequency_MHz(0.0),
          CPU3_Utilization_Percent(0.0), CPU3_Frequency_MHz(0.0),
          CPU4_Utilization_Percent(0.0), CPU4_Frequency_MHz(0.0),
          EMC_Frequency_Percent(0.0), GR3D_Frequency_Percent(0.0),
          PLL_Temperature_C(0.0), CPU_Temperature_C(0.0), PMIC_Temperature_C(0.0),
          GPU_Temperature_C(0.0), AO_Temperature_C(0.0), Thermal_Temperature_C(0.0) {}
};

/**
 * @struct SystemMetricsSnapshot
 * @brief Aggregated snapshot of all system metrics at a single point in time.
 */
struct SystemMetricsSnapshot {
    std::chrono::system_clock::time_point timestamp;
    CameraStats cameraStats;
    AlgorithmStats algorithmStats;
    DisplayStats displayStats;
    JetsonNanoInfo socInfo;
    PowerStats powerStats;
    double joulesPerFrame = 0.0;     // Energy per frame (joules)
    double endToEndLatencyMs = 0.0;  // End-to-end latency (ms)

    SystemMetricsSnapshot() = default;
    SystemMetricsSnapshot(std::chrono::system_clock::time_point ts)
        : timestamp(ts), cameraStats(ts), algorithmStats(ts), displayStats(ts), socInfo(ts), powerStats() {
        powerStats.timestamp = ts; // PowerStats has no timestamp constructor, set manually
    }
};

//#endif // ALL_MODULES_STATCS_H