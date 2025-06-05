// CameraConfig.h
#pragma once
#include <string>
#include <cstdint>       // for uint32_t
//#include <linux/videodev2.h> // for V4L2_PIX_FMT_* macros


/*
Install the Library if Missing
On Ubuntu, the V4L2 development headers/libraries are typically in the package libv4l-dev. You can install it with:

bash
Copy code
sudo apt-get update
sudo apt-get install libv4l-dev
Then rebuild. That should make sure /usr/lib/x86_64-linux-gnu/libv4l2.so is present.

*/


#ifdef __linux__
#include <linux/videodev2.h>
#else
// Define fallback V4L2 constants for non-Linux systems
#define V4L2_PIX_FMT_YUYV 0x56595559 // 'YUYV'
#define V4L2_PIX_FMT_MJPEG 0x47504A4D // 'MJPG'
#endif


/**
 * @brief Enum that wraps V4L2 pixel format codes
 */
enum class PixelFormat : uint32_t {
    YUYV = V4L2_PIX_FMT_YUYV, 
    MJPG = V4L2_PIX_FMT_MJPEG,
    // Add more as needed...
};

/**
 * @brief Example configuration struct for a camera (or other data source).
 *        Adjust fields to your needs (resolution, pixel format, FPS, etc.).
 */


/**
 * @brief Configuration structure for a camera device.
 */

struct CameraConfig {
    int width ;//=320;     //640;               ///< Desired capture width (pixels)
    int height ;//=240;    // 480;             ///< Desired capture height (pixels)
    int fps ;//= 30;                 ///< Desired frames per second
    //std::string pixelFormat = "YUYV";  ///< (e.g., "YUYV", "MJPEG", "RGB24").
    //uint32_t pixelFormat;  // Now stores V4L2 fourcc codes directly
    PixelFormat pixelFormat= PixelFormat::YUYV; // V4L2_PIX_FMT_YUYV; ///< Pixel format (e.g., YUYV, MJPEG)

    // Optionally add a constructor to make usage simpler:
  // Optional constructor
    CameraConfig(int w = 640, int h = 480, int f = 30, PixelFormat pf = PixelFormat::YUYV)
        : width(w), height(h), fps(f), pixelFormat(pf)
    {}
    // Validation method
    bool validate() const {
        if (width <= 0 || height <= 0) return false;
        if (fps <= 0) return false;
        return true;
    }
};
