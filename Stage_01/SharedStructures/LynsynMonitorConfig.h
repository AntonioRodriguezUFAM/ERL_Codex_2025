// LynsynMonitorConfig.h
// LynsynMonitorConfig.h

// Stage_01/SharedStructures/LynsynMonitorConfig.h
#ifndef LYNSYN_MONITOR_CONFIG_H
#define LYNSYN_MONITOR_CONFIG_H

//#pragma once
#include <string>
#include <cstdint>
#include <fstream>
#include <filesystem>

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

/**
 * @struct LynsynMonitorConfig
 * @brief Configuration for Lynsyn power monitoring.
 * @brief Structure holding Lynsyn-related configuration settings,
 *        e.g. which sampling mode, duration, or output file, etc.
 */
struct LynsynMonitorConfig {
     /**
     * An optional output path if you want to do direct CSV logging from this module.
     * If empty, no direct logging is done here.
     */
    std::string outputCSV;    ///< Path to CSV file for logging power samples
      /**
     * If true, we do "period sampling" (time-based).
     * If false, do some other approach or BP-based sampling. 
     */
    bool periodSampling = true; ///< True for time-based sampling, false for breakpoint-based

    /**
     * If set > 0, we do a time-based sampling for `durationSec`.
     * Or if set to 0, it might wait for some breakpoint-based approach, etc.
     */
    double durationSec = 1.0;  ///<   aS,ALSPAs Sampling duration in seconds for period sampling
     /**
     * Which cores to sample for program counters
     * e.g. a bitmask: 0 => no PC sampling
     */
    uint32_t coreMask = 0xF;   ///< Bitmask for cores to monitor (e.g., 0xF for all 4 cores)
    

    /**
     * (Optional) If you want to read from a start/end BP addresses:
     */
    uint64_t startBreakpoint = 0; ///< Start address for breakpoint sampling
    uint64_t endBreakpoint = 0;   ///< End address for breakpoint sampling

    // Validation method
    bool validate() const {
        if (!outputCSV.empty()) {
            // Check if parent directory exists
            fs::path p(outputCSV);
            fs::path parent = p.parent_path();

            if (!parent.empty() && !fs::exists(parent)) {
                return false;
            }
        }

        if (periodSampling && durationSec <= 0) return false;
        if (!periodSampling && (startBreakpoint == 0 || endBreakpoint == 0)) return false;
        if (coreMask == 0) return false;
        return true;
    }
};

#endif // LYNSYN_MONITOR_CONFIG_H

