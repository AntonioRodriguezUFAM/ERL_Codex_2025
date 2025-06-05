
// //DataConcrete_new.h

// #pragma once

// #include "../Interfaces/IData.h"
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/SharedQueue.h"

// #include "../SharedStructures/CameraConfig.h"
// #include "../../Stage_02/Logger/PerformanceLogger.h"  // If you have a logger
// #include <vector>
// #include <atomic>
// #include <mutex>
// #include <thread>
// #include <functional>
// #include <spdlog/spdlog.h>
// #include <linux/videodev2.h>
// #include <sys/mman.h>
// #include <fcntl.h>
// #include <sys/ioctl.h>
// #include <unistd.h>
// #include <cerrno>
// #include <cstring>
// #include <chrono>
// #include <fmt/format.h>

// /**
//  * @class DataConcrete
//  * @brief Manages camera device (V4L2) for capturing frames and pushing them to shared queues.
//  *
//  * Two SharedQueue<FrameData> are injected:
//  * - algoQueue_ (frames for the algorithm)
//  * - displayOrigQueue_ (original frames for display)
//  *
//  * Internally manages buffer states for safe deque/queue operations.
//  */
// class DataConcrete : public IData {
// public:
//     DataConcrete(std::shared_ptr<SharedQueue<FrameData>> algoQueue,
//                  std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue);

//     ~DataConcrete() override;

//     bool openDevice(const std::string& path) override;
//     bool configure(const CameraConfig& config) override;
//     bool startStreaming() override;
//     bool stopStreaming() override;

//     bool dequeFrame(void*& dataPtr, size_t& sizeBytes, size_t& bufferIndex) override;
//     bool queueBuffer(size_t bufferIndex) override;

//     void setErrorCallback(std::function<void(const std::string&)>) override;
//     bool isStreaming() const override;

//     void pushCameraMetrics() override;
//     double getLastFPS() override;
//     int    getQueueSize() const override;

//     void closeDevice();
//     void resetDevice();

// private:
//     enum BufferState { AVAILABLE, QUEUED, DEQUEUED, PROCESSING, IN_USE };
//     struct Buffer {
//         void* start = nullptr;
//         size_t length = 0;
//         BufferState state = AVAILABLE;
//         Buffer() = default;
//     };

//     bool initializeBuffers();
//     bool queueBufferInternal(size_t index);
//     void unmapBuffers();
//     void reportError(const std::string& msg);

//     int  fd_           = -1;
//     bool streaming_    = false;
//     bool configured_   = false;
//     std::string devicePath_;

//     std::vector<Buffer> buffers_;

//     CameraConfig cameraConfig_;
//     std::function<void(const std::string&)> errorCallback_;

//     mutable std::mutex mutex_;
//     mutable std::mutex fpsMutex_;
//     mutable double lastFPS_ = 0.0;
//     mutable std::atomic<int> framesDequeued_{0};

//     mutable std::chrono::steady_clock::time_point lastUpdateTime_;
//     std::atomic<int> framesQueued_{0};

//     // Shared Queues for pipeline
//     std::shared_ptr<SharedQueue<FrameData>> algoQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue_;

//     int NUM_BUFFERS = 4;
// };

// // --------------------- Implementation --------------------- //

// inline DataConcrete::DataConcrete(std::shared_ptr<SharedQueue<FrameData>> algoQueue,
//                                   std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue)
//     : fd_(-1)
//     , streaming_(false)
//     , configured_(false)
//     , lastFPS_(0.0)
//     , algoQueue_(algoQueue)
//     , displayOrigQueue_(displayOrigQueue)
// {
//     lastUpdateTime_ = std::chrono::steady_clock::now();
//     spdlog::debug("[DataConcrete] Constructor complete.");
// }

// inline DataConcrete::~DataConcrete() {
//     if (streaming_) stopStreaming();
//     if (fd_ >= 0)   closeDevice();
//     spdlog::debug("[DataConcrete] Destructor complete.");
// }

// inline bool DataConcrete::openDevice(const std::string& path) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (fd_ != -1) {
//         spdlog::warn("[DataConcrete] Device already open. Closing previous device.");
//         closeDevice();
//     }
//     fd_ = ::open(path.c_str(), O_RDWR);
//     if (fd_ < 0) {
//         reportError("Failed to open " + path + ": " + std::strerror(errno));
//         return false;
//     }
//     devicePath_ = path;
//     spdlog::info("[DataConcrete] Device opened: {}", path);
//     return true;
// }

// inline bool DataConcrete::configure(const CameraConfig& config) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (fd_ < 0) {
//         reportError("Device not opened; cannot configure.");
//         return false;
//     }

//     if (configured_ || !buffers_.empty()) {
//         spdlog::info("[DataConcrete] Re-configuring device. Unmapping old buffers.");
//         unmapBuffers();
//     }
//     cameraConfig_ = config;

//     struct v4l2_format fmt = {};
//     fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     fmt.fmt.pix.width = config.width;
//     fmt.fmt.pix.height = config.height;
//     fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
//     fmt.fmt.pix.field = V4L2_FIELD_NONE;

//     if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
//         reportError("Error setting image format: " + std::string(strerror(errno)));
//         return false;
//     }

//     struct v4l2_format currentFmt = {};
//     currentFmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     if (ioctl(fd_, VIDIOC_G_FMT, &currentFmt) < 0) {
//         reportError("Failed to get camera format after S_FMT: " + std::string(strerror(errno)));
//         return false;
//     }

//     if (currentFmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
//         spdlog::warn("[DataConcrete] Format mismatch: expected YUYV (0x{:X}), got 0x{:X}",
//                      static_cast<unsigned>(V4L2_PIX_FMT_YUYV),
//                      static_cast<unsigned>(currentFmt.fmt.pix.pixelformat));
//     } else {
//         spdlog::info("[DataConcrete] Confirmed camera format is YUYV (0x{:X}).",
//                      static_cast<unsigned>(currentFmt.fmt.pix.pixelformat));
//     }

//     struct v4l2_requestbuffers req = {};
//     req.count = NUM_BUFFERS;
//     req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     req.memory = V4L2_MEMORY_MMAP;
//     if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
//         reportError("Error requesting buffers: " + std::string(strerror(errno)));
//         return false;
//     }

//     if (!initializeBuffers()) {
//         return false;
//     }

//     configured_ = true;
//     spdlog::info("[DataConcrete] Device configured ({}x{}).", config.width, config.height);
//     return true;
// }

// inline bool DataConcrete::initializeBuffers() {
//     buffers_.clear();
//     buffers_.resize(NUM_BUFFERS);

//     for (int i = 0; i < NUM_BUFFERS; ++i) {
//         struct v4l2_buffer buf = {};
//         buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//         buf.memory = V4L2_MEMORY_MMAP;
//         buf.index = i;

//         if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
//             reportError("VIDIOC_QUERYBUF failed for buffer " + std::to_string(i));
//             return false;
//         }

//         buffers_[i].length = buf.length;
//         buffers_[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
//         if (buffers_[i].start == MAP_FAILED) {
//             reportError("mmap failed for buffer " + std::to_string(i));
//             return false;
//         }
//         buffers_[i].state = AVAILABLE;
//         spdlog::debug("[DataConcrete] Buffer {} mapped, size: {}", i, buf.length);
//     }
//     return true;
// }

// inline bool DataConcrete::startStreaming() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (streaming_) {
//         spdlog::warn("[DataConcrete] Already streaming.");
//         return true;
//     }
//     if (!configured_ || buffers_.empty()) {
//         reportError("Device not configured or no buffers allocated.");
//         return false;
//     }

//     for (size_t i = 0; i < buffers_.size(); ++i) {
//         if (!queueBufferInternal(i)) {
//             spdlog::error("[DataConcrete] Failed to queue buffer {}", i);
//             return false;
//         }
//     }

//     enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
//         reportError("VIDIOC_STREAMON failed: " + std::string(strerror(errno)));
//         return false;
//     }
//     streaming_ = true;
//     spdlog::info("[DataConcrete] Streaming started with {} buffers", buffers_.size());
//     return true;
// }

// inline bool DataConcrete::stopStreaming() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (!streaming_) {
//         spdlog::warn("[DataConcrete] Not streaming; nothing to stop.");
//         return true;
//     }
//     enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     if (ioctl(fd_, VIDIOC_STREAMOFF, &type) < 0) {
//         reportError("Error stopping stream: " + std::string(strerror(errno)));
//     }
//     streaming_ = false;
//     spdlog::info("[DataConcrete] Streaming stopped.");
//     return true;
// }

// inline bool DataConcrete::dequeFrame(void*& dataPtr, size_t& sizeBytes, size_t& bufferIndex)
// {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (!streaming_) {
//         reportError("[DataConcrete] Cannot dequeue frame: Not streaming.");
//         return false;
//     }

//     struct v4l2_buffer buf = {};
//     buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     buf.memory = V4L2_MEMORY_MMAP;

//     if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
//         if (errno == EAGAIN) {
//             spdlog::debug("[DataConcrete] No frame available (EAGAIN).");
//             return false;
//         } else {
//             reportError(std::string("[DataConcrete] Error dequeuing buffer: ") + strerror(errno));
//             return false;
//         }
//     }

//     if (buf.index >= buffers_.size()) {
//         reportError("[DataConcrete] Dequeued buffer index out of range: " + std::to_string(buf.index));
//         return false;
//     }

//     auto& buffer = buffers_[buf.index];
//     if (buffer.state != QUEUED) {
//         reportError("[DataConcrete] Buffer " + std::to_string(buf.index) + " is not in QUEUED state.");
//         return false;
//     }

//     bufferIndex = buf.index;
//     buffer.state = DEQUEUED;
//     spdlog::debug("[DataConcrete] Buffer {} dequeued.", bufferIndex);

//     // Attempt automatic requeue.
//     if (!queueBuffer(bufferIndex)) {
//         buffer.state = AVAILABLE;
//         reportError("Failed to requeue buffer");
//         return false;
//     }

//     dataPtr = buffer.start;
//     sizeBytes = buf.bytesused;
//     framesDequeued_++;
//     spdlog::debug("[DataConcrete] Frame captured: buffer #{}, size={}, frame#={}",
//                   bufferIndex, sizeBytes, framesDequeued_.load());

//     // Copy frame data to a FrameData object.
//     FrameData frame;
//     frame.dataVec.resize(sizeBytes);
//     std::memcpy(frame.dataVec.data(), dataPtr, sizeBytes);
//     frame.size = sizeBytes;
//     frame.width = cameraConfig_.width;
//     frame.height = cameraConfig_.height;
//     frame.frameNumber = framesDequeued_.load();

//         // Validate Queue Initialization:
//     if (!displayOrigQueue_ || !algoQueue_) {
//         reportError("Queues not initialized in DataConcrete!");
//         return false;
//     }

//     // Push the frame into both pipelines.
//     if (displayOrigQueue_) {
//         displayOrigQueue_->push(frame);
//         spdlog::debug("[DataConcrete] Frame pushed to display queue. Queue size now: {}",
//                       displayOrigQueue_->size());
//     }
//     if (algoQueue_) {
//         algoQueue_->push(frame);
//         spdlog::debug("[DataConcrete] Frame pushed to algorithm queue. Queue size now: {}",
//                       algoQueue_->size());
//     }

//     spdlog::debug("[DataConcrete] Successfully dequeued and processed frame from buffer #{}", bufferIndex);
//     if (static_cast<uint32_t>(cameraConfig_.pixelFormat) != V4L2_PIX_FMT_YUYV) {
//         spdlog::warn("[DataConcrete] Camera format mismatch: expected YUYV (0x{:X})", 
//                      static_cast<unsigned>(V4L2_PIX_FMT_YUYV));
//     }
//     return true;
// }

// inline bool DataConcrete::queueBuffer(size_t bufferIndex) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (bufferIndex >= buffers_.size()) {
//         reportError("Invalid buffer index: " + std::to_string(bufferIndex));
//         return false;
//     }
//     return queueBufferInternal(bufferIndex);
// }

// inline bool DataConcrete::queueBufferInternal(size_t index) {
//     if (index >= buffers_.size()) {
//         reportError("queueBufferInternal: Index out of range: " + std::to_string(index));
//         return false;
//     }
//     auto& buffer = buffers_[index];
//     if (buffer.state == QUEUED) {
//         reportError("Buffer " + std::to_string(index) + " is already QUEUED.");
//         return false;
//     }
//     struct v4l2_buffer buf = {};
//     buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     buf.memory = V4L2_MEMORY_MMAP;
//     buf.index = index;
//     if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
//         buffer.state = AVAILABLE;
//         reportError("Error queueing buffer index " + std::to_string(index) + ": " + std::string(strerror(errno)));
//         return false;
//     }
//     buffer.state = QUEUED;
//     spdlog::debug("[DataConcrete] Buffer {} queued.", index);
//     return true;
// }

// inline void DataConcrete::unmapBuffers() {
//     for (auto& buf : buffers_) {
//         if (buf.start && buf.start != MAP_FAILED) {
//             munmap(buf.start, buf.length);
//         }
//         buf.start = nullptr;
//         buf.length = 0;
//         buf.state = AVAILABLE;
//     }
//     buffers_.clear();
//     spdlog::debug("[DataConcrete] All buffers unmapped and cleared.");
// }

// inline void DataConcrete::reportError(const std::string& msg) {
//     if (errorCallback_) {
//         errorCallback_(msg);
//     } else {
//         spdlog::error("[DataConcrete] {}", msg);
//     }
// }

// inline void DataConcrete::setErrorCallback(std::function<void(const std::string&)> callback) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     errorCallback_ = std::move(callback);
// }

// inline bool DataConcrete::isStreaming() const {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return streaming_;
// }

// inline void DataConcrete::pushCameraMetrics() {
//     double fps = getLastFPS();
//     PerformanceLogger::getInstance().pushCameraStats(0, fps, framesQueued_.load());
// }

// inline double DataConcrete::getLastFPS() {
//     std::lock_guard<std::mutex> lock(fpsMutex_);
//     auto now = std::chrono::steady_clock::now();
//     double duration = std::chrono::duration<double>(now - lastUpdateTime_).count();
//     if (duration > 1.0) {
//         int count = framesDequeued_.exchange(0);
//         lastFPS_ = static_cast<double>(count) / duration;
//         lastUpdateTime_ = now;
//     }
//     return lastFPS_;
// }

// inline int DataConcrete::getQueueSize() const {
//     return framesQueued_.load();
// }

// inline void DataConcrete::closeDevice() {
//     unmapBuffers();
//     if (fd_ != -1) {
//         ::close(fd_);
//         fd_ = -1;
//         spdlog::info("[DataConcrete] Device closed.");
//     }
// }

// inline void DataConcrete::resetDevice() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (streaming_) {
//         stopStreaming();
//     }
//     closeDevice();
//     spdlog::info("[DataConcrete] Device reset.");
// }

//  =================================================================
//  SystemCaptureFactory_new.h
//  ============================================================

// DataConcrete_new
// (Stage 2: Camera/IData Concrete Class)


//DataConcrete_new.h

//DataConcrete_new.h - Updated Version


// #pragma once

// #include "../Interfaces/IData.h"
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/SharedQueue.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../../Stage_02/Logger/PerformanceLogger.h"  // If you have a logger

// #include <vector>
// #include <atomic>
// #include <mutex>
// #include <thread>
// #include <functional>
// #include <string>
// #include <spdlog/spdlog.h>

// #include <linux/videodev2.h>
// #include <sys/mman.h>
// #include <fcntl.h>
// #include <sys/ioctl.h>
// #include <unistd.h>
// #include <cerrno>
// #include <cstring>
// #include <chrono>

// #include <fmt/format.h> // For fmt::format in error messages

//     // Instead of copying data, pass a pointer to mapped buffer directly:

//     /**
//      * @brief Structure containing raw frame data captured from the camera
//      * Modified to support zero-copy
//      */
//     struct ZeroCopyFrameData {
//         void* dataPtr;       // Pointer to the original buffer
//         size_t size;         // Size of the frame data
//         int width;           // Frame width
//         int height;          // Frame height
//         size_t bufferIndex;  // Index for buffer management
//         int frameNumber;     // Frame sequence number
//         std::chrono::steady_clock::time_point captureTime; // Capture timestamp
//         // Other metadata...
//     };
    


// /* Bufer manager
// * Reference Counting with SharedQueue
// *   Implementation Steps:
// *   Create a BufferManager to track buffer references
// *   Increment reference count when buffer is queued
// *   Decrement when buffer is processed and returned
// */
// class BufferManager {
//     public:
//         using RequeueCallback = std::function<void(size_t)>;
//         BufferManager(RequeueCallback callback) : requeueCallback_(std::move(callback)) {}
    
//         void incrementRef(size_t bufferIndex) {
//             std::lock_guard<std::mutex> lock(mutex_);
//             if (bufferIndex >= refCounts_.size()) refCounts_.resize(bufferIndex + 1, 0);
//             refCounts_[bufferIndex]++;
//         }
    
//         void decrementRef(size_t bufferIndex) {
//             std::lock_guard<std::mutex> lock(mutex_);
//             if (bufferIndex >= refCounts_.size() || refCounts_[bufferIndex] == 0) return;
//             if (--refCounts_[bufferIndex] == 0) requeueCallback_(bufferIndex);
//         }
    
//     private:
//         std::vector<int> refCounts_;
//         std::mutex mutex_;
//         RequeueCallback requeueCallback_;
//     };

// /**
//  * @class DataConcrete
//  * @brief Manages camera device (V4L2) for capturing frames and pushing them to shared queues.
//  *
//  * Two SharedQueue<std::shared_ptr<ZeroCopyFrameData>> are injected:
//  * - algoQueue_       (frames for algorithm processing)
//  * - displayOrigQueue_(original frames for display)
//  *
//  * Internally manages buffer states for safe deque/queue operations, using a circular buffering approach.
//  * Once a frame is dequeued and copied, the buffer is immediately re-queued within the same method.
//  *
//  */


// class DataConcrete : public IData {
// public:
// DataConcrete(
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQueue,
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> displayOrigQueue
// );

//     ~DataConcrete() override;

//     bool openDevice(const std::string& path) override;
//     bool configure(const CameraConfig& config) override;
//     bool startStreaming() override;
//     bool stopStreaming() override;

//     bool dequeFrame(void*& dataPtr, size_t& sizeBytes, size_t& bufferIndex) override;
//     bool dequeFrame() override;
//     bool queueBuffer(size_t bufferIndex) override;

//     void setErrorCallback(std::function<void(const std::string&)>) override;
//     bool isStreaming() const override;

//     void pushCameraMetrics() override; 
//     double getLastFPS() override;
//     int    getQueueSize() const override;

//     // Additional device management methods
//     void closeDevice();
//     void resetDevice();

//     // Added getter functions
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> getDisplayQueue() const {
//         return displayOrigQueue_;
//     }

//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> getProcessingQueue() const {
//         return algoQueue_;
//     }

// private:
//     /// Buffer states for internal ring buffer logic
//     enum BufferState { AVAILABLE, QUEUED, DEQUEUED, PROCESSING, IN_USE };

//     // Represents one V4L2 buffer mapping
//     struct Buffer {
//         void*       start = nullptr;
//         size_t      length = 0;
//         BufferState state = AVAILABLE;
//         Buffer() = default;
//         std::chrono::steady_clock::time_point captureTime;
//     };

//     // Core V4L2 operations
//     bool initializeBuffers();
//     bool queueBufferInternal(size_t index);
//     void unmapBuffers();
//     void reportError(const std::string& msg);
//     void cleanUpResources();

//     // 3. Buffer Pooling with SharedQueue
//   //  BufferPool bufferPool_(maxFrameSize, 10);

//     // Device state
//     int  fd_           = -1;
//     std::atomic<bool> streaming_{false};
//     bool configured_ = false;
//     std::string devicePath_;
//     CameraConfig cameraConfig_;

//     // Memory-mapped buffers
//     // Buffer management is done with a circular buffer approach
//     std::vector<Buffer> buffers_;
//     mutable std::mutex bufferMutex_;


//     // If we want a configurable number of buffers, We could pass it in via CameraConfig.
//     static constexpr int NUM_BUFFERS = 4;

  
//     std::function<void(const std::string&)> errorCallback_;

//     // Lock for camera streaming operations
//     mutable std::mutex mutex_;

//     // Lock for FPS counters
//     mutable std::mutex fpsMutex_;
//     mutable double lastFPS_ = 0.0;
//     mutable std::atomic<int> framesDequeued_{0};
//     mutable std::chrono::steady_clock::time_point lastUpdateTime_;

//     // For future queue size metrics (not incremented in this snippet)
//     std::atomic<int> framesQueued_{0};

//     // Shared queues for pipeline

//     //BufferManager bufferManager_;
//     //std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQueue_, displayQueue_;

//      // Shared queues for pipeline
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQueue_;
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> displayOrigQueue_;


// };


// // --------------------- Implementation --------------------- //
// inline DataConcrete::DataConcrete(
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQueue,
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> displayOrigQueue
// ) : fd_(-1)
//     , streaming_(false)
//     , configured_(false)
//     , lastFPS_(0.0)
//     , algoQueue_(std::move(algoQueue))
//     , displayOrigQueue_(std::move(displayOrigQueue))
// {
//     lastUpdateTime_ = std::chrono::steady_clock::now();
//     spdlog::debug("[DataConcrete] Constructor complete.");
// }

// inline DataConcrete::~DataConcrete() {
//     if (streaming_) {
//         stopStreaming();
//     }
//     if (fd_ >= 0) {
//         closeDevice();
//     }
//     spdlog::debug("[DataConcrete] Destructor complete.");
// }

// inline bool DataConcrete::openDevice(const std::string& path) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (fd_ != -1) {
//         spdlog::warn("[DataConcrete] Device already open. Closing previous device.");
//         closeDevice();
//     }
//     fd_ = ::open(path.c_str(), O_RDWR);
//     if (fd_ < 0) {
//         reportError(fmt::format("Failed to open {}: {}", path, std::strerror(errno)));
//         return false;
//     }
//     devicePath_ = path;
//     spdlog::info("[DataConcrete] Device opened: {}", path);
//     return true;
// }

// inline bool DataConcrete::configure(const CameraConfig& config) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (fd_ < 0) {
//         reportError("Device not opened; cannot configure.");
//         return false;
//     }
//     // If already configured, unmap old buffers
//     if (configured_ || !buffers_.empty()) {
//         spdlog::info("[DataConcrete] Re-configuring device. Unmapping old buffers.");
//         unmapBuffers();
//     }

//     cameraConfig_ = config;

//     // Set YUYV format
//     struct v4l2_format fmt = {};
//     fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     fmt.fmt.pix.width       = config.width;
//     fmt.fmt.pix.height      = config.height;
//     fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
//     fmt.fmt.pix.field       = V4L2_FIELD_NONE;

//     if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
//         reportError(fmt::format("VIDIOC_S_FMT failed: {}", std::strerror(errno)));
//         return false;
//     }

//     // Check driver output
//     struct v4l2_format currentFmt = {};
//     currentFmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     if (ioctl(fd_, VIDIOC_G_FMT, &currentFmt) < 0) {
//         reportError(fmt::format("VIDIOC_G_FMT failed after S_FMT: {}", std::strerror(errno)));
//         return false;
//     }
//     if (currentFmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
//         // This updated version is strict: it fails if not YUYV
//         reportError(fmt::format("Format mismatch after S_FMT: Expected YUYV (0x{:X}), got 0x{:X}",
//                      static_cast<unsigned>(V4L2_PIX_FMT_YUYV),
//                      static_cast<unsigned>(currentFmt.fmt.pix.pixelformat)));
//         return false;
//     } else {
//         spdlog::info("[DataConcrete] Confirmed camera format is YUYV (0x{:X}).",
//                      static_cast<unsigned>(currentFmt.fmt.pix.pixelformat));
//     }

//     // Request buffers
//     struct v4l2_requestbuffers req = {};
//     req.count  = NUM_BUFFERS;
//     req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     req.memory = V4L2_MEMORY_MMAP;
//     if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
//         reportError(fmt::format("VIDIOC_REQBUFS failed: {}", std::strerror(errno)));
//         return false;
//     }

//     if (!initializeBuffers()) {
//         return false;
//     }
//     configured_ = true;
//     spdlog::info("[DataConcrete] Device configured ({}x{}).", config.width, config.height);
//     return true;
// }

// inline bool DataConcrete::initializeBuffers() {
//     buffers_.clear();
//     buffers_.resize(NUM_BUFFERS);

//     for (int i = 0; i < NUM_BUFFERS; ++i) {
//         struct v4l2_buffer buf = {};
//         buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//         buf.memory = V4L2_MEMORY_MMAP;
//         buf.index  = i;

//         if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
//             reportError(fmt::format("VIDIOC_QUERYBUF failed for buffer {}: {}", 
//                                     i, std::strerror(errno)));
//             return false;
//         }

//         buffers_[i].length = buf.length;
//         buffers_[i].start  = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
//                                   MAP_SHARED, fd_, buf.m.offset);
//         if (buffers_[i].start == MAP_FAILED) {
//             reportError(fmt::format("mmap failed for buffer {}: {}", i, std::strerror(errno)));
//             return false;
//         }
//         buffers_[i].state = AVAILABLE;
//         spdlog::debug("[DataConcrete] Buffer {} mapped, size: {}", i, buf.length);
//     }
//     return true;
// }

// inline bool DataConcrete::startStreaming() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (streaming_) {
//         spdlog::warn("[DataConcrete] Already streaming.");
//         return true;
//     }
//     if (!configured_ || buffers_.empty()) {
//         reportError("Device not configured or no buffers allocated.");
//         return false;
//     }

//     // Queue all buffers initially
//     for (size_t i = 0; i < buffers_.size(); ++i) {
//         if (!queueBufferInternal(i)) {
//             spdlog::error("[DataConcrete] Failed to queue buffer {}", i);
//             return false;
//         }
//     }

//     enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
//         reportError(fmt::format("VIDIOC_STREAMON failed: {}", std::strerror(errno)));
//         return false;
//     }
//     streaming_ = true;
//     spdlog::info("[DataConcrete] Streaming started with {} buffers", buffers_.size());
//     return true;
// }

// inline bool DataConcrete::stopStreaming() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (!streaming_) {
//         spdlog::warn("[DataConcrete] Not streaming; nothing to stop.");
//         return true;
//     }
//     enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     if (ioctl(fd_, VIDIOC_STREAMOFF, &type) < 0) {
//         reportError(fmt::format("VIDIOC_STREAMOFF failed: {}", std::strerror(errno)));
//     }
//     streaming_ = false;
//     spdlog::info("[DataConcrete] Streaming stopped.");
//     return true;
// }


// // In DataConcrete::dequeFrame
// inline bool DataConcrete::dequeFrame(void*& dataPtr, size_t& sizeBytes, size_t& bufferIndex) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (!streaming_) {
//         reportError("[DataConcrete] Cannot dequeue frame: Not streaming.");
//         return false;
//     }

//     struct v4l2_buffer buf = {};
//     buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     buf.memory = V4L2_MEMORY_MMAP;

//     if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
//         if (errno == EAGAIN) {
//             spdlog::debug("[DataConcrete] No frame available (EAGAIN).");
//             return false;
//         } else {
//             reportError(fmt::format("VIDIOC_DQBUF failed: {}", std::strerror(errno)));
//             return false;
//         }
//     }
//     if (buf.index >= buffers_.size()) {
//         reportError(fmt::format("Dequeued buffer index out of range: {}", buf.index));
//         return false;
//     }

//     auto& buffer = buffers_[buf.index];
//     if (buffer.state != QUEUED) {
//         reportError(fmt::format("Buffer {} is not in QUEUED state.", buf.index));
//         return false;
//     }

//     bufferIndex = buf.index;
//     buffer.state = DEQUEUED;
//     spdlog::debug("[DataConcrete] Buffer {} dequeued.", bufferIndex);

//     // Attempt automatic requeue: circular buffering
//     if (!queueBuffer(bufferIndex)) {
//         buffer.state = AVAILABLE;
//         reportError("Failed to requeue buffer");
//         return false;
//     }

//     dataPtr   = buffer.start;
//     sizeBytes = buf.bytesused;
//     framesDequeued_++;

//     spdlog::debug("[DataConcrete] Frame captured: buffer #{}, size={}, frame#={}",
//                   bufferIndex, sizeBytes, framesDequeued_.load());


//     // Create FrameData with pointer to buffer - Zero-Copy Implementation:
//     //ZeroCopyFrameData frame { dataPtr, sizeBytes, cameraConfig_.width, cameraConfig_.height, bufferIndex };

//     auto frame = std::make_shared<ZeroCopyFrameData>();
//     frame->dataPtr = buffer.start;
//     frame->size = buf.bytesused;
//     frame->width = cameraConfig_.width;
//     frame->height = cameraConfig_.height;
//     frame->frameNumber = framesDequeued_.load();
//     frame->bufferIndex = bufferIndex; // Store buffer index for later reference
//     frame->captureTime = std::chrono::steady_clock::now();
                  
//     // Validate queue pointers
//     if (!displayOrigQueue_ || !algoQueue_) {
//         reportError("Queues not initialized in DataConcrete!");
//         return false;
//     }

//     // Push to both queues

//     // Push zero-copy frame data into queues directly
//     if (displayOrigQueue_) {
//         displayOrigQueue_->push(frame);
//         spdlog::debug("Zero-copy frame pushed to display queue. Queue size now: {}",
//                      displayOrigQueue_->size());
//     }

//     if (algoQueue_) {
//         algoQueue_->push(frame);
//         spdlog::debug("Zero-copy frame pushed to algorithm queue. Queue size now: {}",
//                       algoQueue_->size());
//     }


//     spdlog::debug("[DataConcrete] Successfully dequeued and processed frame from buffer #{}", bufferIndex);


//     if (static_cast<uint32_t>(cameraConfig_.pixelFormat) != V4L2_PIX_FMT_YUYV) {
//         spdlog::warn("[DataConcrete] Camera format mismatch: expected YUYV (0x{:X})", 
//                      static_cast<unsigned>(V4L2_PIX_FMT_YUYV));
//     }
//     return true;

// }

// // Implementation of dequeFrame using zero-copy
// inline bool DataConcrete::dequeFrame() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (!streaming_) {
//         reportError("Not streaming; cannot dequeue frame.");
//         return false;
//     }

//     v4l2_buffer buf{};
//     buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     buf.memory = V4L2_MEMORY_MMAP;

//     if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
//         if (errno != EAGAIN) {
//             reportError(fmt::format("VIDIOC_DQBUF error: {}", strerror(errno)));
//         }
//         return false;
//     }

//     if (buf.index >= buffers_.size()) {
//         reportError(fmt::format("Buffer index out of range: {}", buf.index));
//         return false;
//     }

//     buffers_[buf.index].state = DEQUEUED;

//     ZeroCopyFrameData frame{
//         buffers_[buf.index].start,
//         buf.bytesused,
//         cameraConfig_.width,
//         cameraConfig_.height,
//         buf.index
//     };

//     if (displayOrigQueue_) displayOrigQueue_->push(frame);
//     if (algoQueue_) algoQueue_->push(frame);

//     buffers_[buf.index].state = QUEUED;
//     ioctl(fd_, VIDIOC_QBUF, &buf);

//     framesDequeued_++;

//     return true;
// }


// inline bool DataConcrete::queueBuffer(size_t bufferIndex) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (bufferIndex >= buffers_.size()) {
//         reportError(fmt::format("Invalid buffer index: {}", bufferIndex));
//         return false;
//     }
//     return queueBufferInternal(bufferIndex);
// }

// inline bool DataConcrete::queueBufferInternal(size_t index) {
//     if (index >= buffers_.size()) {
//         reportError(fmt::format("queueBufferInternal: Index out of range: {}", index));
//         return false;
//     }
//     auto& buffer = buffers_[index];
//     if (buffer.state == QUEUED) {
//         reportError(fmt::format("Buffer {} is already QUEUED.", index));
//         return false;
//     }

//     struct v4l2_buffer buf = {};
//     buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     buf.memory = V4L2_MEMORY_MMAP;
//     buf.index  = index;
//     if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
//         reportError(fmt::format("VIDIOC_QBUF failed for buffer {}: {}", 
//                                 index, std::strerror(errno)));
//         return false;
//     }
//     buffer.state = QUEUED;
//     spdlog::debug("[DataConcrete] Buffer {} queued.", index);
//     return true;
// }

// inline void DataConcrete::unmapBuffers() {
//     // Make sure streaming is stopped before unmapping (done in destructor or resetDevice).
//     for (auto& buf : buffers_) {
//         if (buf.start && buf.start != MAP_FAILED) {
//             munmap(buf.start, buf.length);
//         }
//         buf.start  = nullptr;
//         buf.length = 0;
//         buf.state  = AVAILABLE;
//     }
//     buffers_.clear();
//     spdlog::debug("[DataConcrete] All buffers unmapped and cleared.");
// }

// inline void DataConcrete::reportError(const std::string& msg) {
//     if (errorCallback_) {
//         errorCallback_(msg);
//     } else {
//         spdlog::error("[DataConcrete] {}", msg);
//     }
// }

// inline void DataConcrete::setErrorCallback(std::function<void(const std::string&)> callback) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     errorCallback_ = std::move(callback);
// }

// inline bool DataConcrete::isStreaming() const {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return streaming_;
// }

// inline void DataConcrete::pushCameraMetrics() {
//     double fps = getLastFPS();
//     // For a real system, you might pass more data to PerformanceLogger
//     PerformanceLogger::getInstance().pushCameraStats(0, fps, framesQueued_.load());
// }

// inline double DataConcrete::getLastFPS() {
//     std::lock_guard<std::mutex> lock(fpsMutex_);
//     auto now      = std::chrono::steady_clock::now();
//     double elapsedSec = std::chrono::duration<double>(now - lastUpdateTime_).count();
//     if (elapsedSec > 1.0) {
//         int count = framesDequeued_.exchange(0);
//         lastFPS_  = static_cast<double>(count) / elapsedSec;
//         lastUpdateTime_ = now;
//     }
//     return lastFPS_;
// }

// inline int DataConcrete::getQueueSize() const {
//     // By default, not incremented anywhere; you can do so in `dequeFrame` or when pushing frames
//     return framesQueued_.load();
// }

// inline void DataConcrete::closeDevice() {
//     unmapBuffers();
//     if (fd_ != -1) {
//         ::close(fd_);
//         fd_ = -1;
//         spdlog::info("[DataConcrete] Device closed.");
//     }
// }

// inline void DataConcrete::resetDevice() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (streaming_) {
//         stopStreaming();
//     }
//     closeDevice();
//     spdlog::info("[DataConcrete] Device reset.");
// }


///=====================Working code Final Version ==============================

// // DataConcrete_new.h

// #pragma once

// #include "../Interfaces/IData.h"
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/SharedQueue.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/ZeroCopyFrameData.h"
// #include "../../Stage_02/Logger/PerformanceLogger.h"  // If you have a logger

// #include <vector>
// #include <atomic>
// #include <mutex>
// #include <thread>
// #include <functional>
// #include <string>
// #include <spdlog/spdlog.h>

// #include <linux/videodev2.h>
// #include <sys/mman.h>
// #include <fcntl.h>
// #include <sys/ioctl.h>
// #include <unistd.h>
// #include <cerrno>
// #include <cstring>
// #include <chrono>

// #include <fmt/format.h> // For fmt::format in error messages

// // /**
// //  * @brief Structure containing raw frame data captured from the camera
// //  * Modified to support zero-copy
// //  */
// // struct ZeroCopyFrameData {
// //     void* dataPtr;       // Pointer to the original buffer
// //     size_t size;         // Size of the frame data
// //     int width;           // Frame width
// //     int height;          // Frame height
// //     size_t bufferIndex;  // Index for buffer management
// //     int frameNumber;     // Frame sequence number
// //     std::chrono::steady_clock::time_point captureTime; // Capture timestamp
// //     // Other metadata...
// // };

// /**
//  * @class DataConcrete
//  * @brief Manages camera device (V4L2) for capturing frames and pushing them to shared queues.
//  *
//  * Two SharedQueue<std::shared_ptr<ZeroCopyFrameData>> are injected:
//  * - algoQueue_       (frames for algorithm processing)
//  * - displayOrigQueue_(original frames for display)
//  *
//  * Internally manages buffer states for safe deque/queue operations, using a circular buffering approach.
//  * Once a frame is dequeued and copied, the buffer is immediately re-queued within the same method.
//  *
//  */




 
// class DataConcrete : public IData {
// public:
//     DataConcrete(
//         std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQueue,
//         std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> displayOrigQueue
//     );

//     ~DataConcrete() override;

//     bool openDevice(const std::string& path) override;
//     bool configure(const CameraConfig& config) override;
//     bool startStreaming() override;
//     bool stopStreaming() override;

//     //bool dequeFrame(void*& dataPtr, size_t& sizeBytes, size_t& bufferIndex) override;
//     bool dequeFrame() override;
//     bool queueBuffer(size_t bufferIndex) override;

//     void setErrorCallback(std::function<void(const std::string&)>) override;
//     bool isStreaming() const override;

//     void pushCameraMetrics() override; 
//     double getLastFPS() override;
//     int    getQueueSize() const override;

//     // Additional device management methods
//     void closeDevice();
//     void resetDevice();

//     // Added getter functions
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> getDisplayQueue() const {
//         return displayOrigQueue_;
//     }

//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> getProcessingQueue() const {
//         return algoQueue_;
//     }

// private:
//     /// Buffer states for internal ring buffer logic
//     enum BufferState { AVAILABLE, QUEUED, DEQUEUED, PROCESSING, IN_USE };

//     // Represents one V4L2 buffer mapping
//     struct Buffer {
//         void*       start = nullptr;
//         size_t      length = 0;
//         BufferState state = AVAILABLE;
//         Buffer() = default;
//         //enum State {AVAILABLE, QUEUED, DEQUEUED} state;
//         std::chrono::steady_clock::time_point captureTime;
//     };

//     // Core V4L2 operations
//     bool initializeBuffers();
//     bool queueBufferInternal(size_t index);
//     void unmapBuffers();
//     void reportError(const std::string& msg);
//     void cleanUpResources();

//     // Device state
//     int  fd_           = -1;
//     std::atomic<bool> streaming_{false};
//     bool configured_ = false;
//     std::string devicePath_;
//     CameraConfig cameraConfig_;

//     // Memory-mapped buffers
//     // Buffer management is done with a circular buffer approach
//     std::vector<Buffer> buffers_;
//     mutable std::mutex bufferMutex_;


//     // If we want a configurable number of buffers, We could pass it in via CameraConfig.
//     static constexpr int NUM_BUFFERS = 4;

  
//     std::function<void(const std::string&)> errorCallback_;

//     // Lock for camera streaming operations
//     mutable std::mutex mutex_;

//     // Lock for FPS counters
//     mutable std::mutex fpsMutex_;
//     mutable double lastFPS_ = 0.0;
//     mutable std::atomic<int> framesDequeued_{0};
//     mutable std::chrono::steady_clock::time_point lastUpdateTime_;

//     // For future queue size metrics (not incremented in this snippet)
//     std::atomic<int> framesQueued_{0};

//     // Shared queues for pipeline
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQueue_;
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> displayOrigQueue_;
// };

// // --------------------- Implementation --------------------- //
// inline DataConcrete::DataConcrete(
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQueue,
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> displayOrigQueue
// ) : fd_(-1)
//     , streaming_(false)
//     , configured_(false)
//     // Initialization code
//     , lastFPS_(0.0)
//     , algoQueue_(std::move(algoQueue))
//     , displayOrigQueue_(std::move(displayOrigQueue))
// {
//     lastUpdateTime_ = std::chrono::steady_clock::now();
//     spdlog::debug("[DataConcrete] Constructor complete.");
// }

// inline DataConcrete::~DataConcrete() {
//     if (streaming_) {
//         stopStreaming();
//     }
//     if (fd_ >= 0) {
//         closeDevice();
//     }
//     spdlog::debug("[DataConcrete] Destructor complete.");
// }

// inline bool DataConcrete::openDevice(const std::string& path) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (fd_ != -1) {
//         spdlog::warn("[DataConcrete] Device already open. Closing previous device.");
//         closeDevice();
//     }
//     fd_ = ::open(path.c_str(), O_RDWR);
//     if (fd_ < 0) {
//         reportError(fmt::format("Failed to open {}: {}", path, std::strerror(errno)));
//         return false;
//     }
//     devicePath_ = path;
//     spdlog::info("[DataConcrete] Device opened: {}", path);
//     return true;
// }

// inline bool DataConcrete::configure(const CameraConfig& config) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (fd_ < 0) {
//         reportError("Device not opened; cannot configure.");
//         return false;
//     }
//     // If already configured, unmap old buffers
//     if (configured_ || !buffers_.empty()) {
//         spdlog::info("[DataConcrete] Re-configuring device. Unmapping old buffers.");
//         unmapBuffers();
//     }

//     cameraConfig_ = config;

//     // Set YUYV format
//     struct v4l2_format fmt = {};
//     fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     fmt.fmt.pix.width       = config.width;
//     fmt.fmt.pix.height      = config.height;
//     fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
//     fmt.fmt.pix.field       = V4L2_FIELD_NONE;

//     if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
//         reportError(fmt::format("VIDIOC_S_FMT failed: {}", std::strerror(errno)));
//         return false;
//     }

//     // Check driver output
//     struct v4l2_format currentFmt = {};
//     currentFmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     if (ioctl(fd_, VIDIOC_G_FMT, &currentFmt) < 0) {
//         reportError(fmt::format("VIDIOC_G_FMT failed after S_FMT: {}", std::strerror(errno)));
//         return false;
//     }
//     if (currentFmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
//         // This updated version is strict: it fails if not YUYV
//         reportError(fmt::format("Format mismatch after S_FMT: Expected YUYV (0x{:X}), got 0x{:X}",
//                      static_cast<unsigned>(V4L2_PIX_FMT_YUYV),
//                      static_cast<unsigned>(currentFmt.fmt.pix.pixelformat)));
//         return false;
//     } else {
//         spdlog::info("[DataConcrete] Confirmed camera format is YUYV (0x{:X}).",
//                      static_cast<unsigned>(currentFmt.fmt.pix.pixelformat));
//     }

//     // Request buffers
//     struct v4l2_requestbuffers req = {};
//     req.count  = NUM_BUFFERS;
//     req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     req.memory = V4L2_MEMORY_MMAP;
//     if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
//         reportError(fmt::format("VIDIOC_REQBUFS failed: {}", std::strerror(errno)));
//         return false;
//     }

//     if (!initializeBuffers()) {
//         return false;
//     }
//     configured_ = true;
//     spdlog::info("[DataConcrete] Device configured ({}x{}).", config.width, config.height);
//     return true;
// }

// inline bool DataConcrete::initializeBuffers() {
//     buffers_.clear();
//     buffers_.resize(NUM_BUFFERS);

//     for (int i = 0; i < NUM_BUFFERS; ++i) {
//         struct v4l2_buffer buf = {};
//         buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//         buf.memory = V4L2_MEMORY_MMAP;
//         buf.index  = i;

//         if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
//             reportError(fmt::format("VIDIOC_QUERYBUF failed for buffer {}: {}", 
//                                     i, std::strerror(errno)));
//             return false;
//         }

//         buffers_[i].length = buf.length;
//         buffers_[i].start  = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
//                                   MAP_SHARED, fd_, buf.m.offset);
//         if (buffers_[i].start == MAP_FAILED) {
//             reportError(fmt::format("mmap failed for buffer {}: {}", i, std::strerror(errno)));
//             return false;
//         }
//         buffers_[i].state = AVAILABLE;
//         spdlog::debug("[DataConcrete] Buffer {} mapped, size: {}", i, buf.length);
//     }
//     return true;
// }

// inline bool DataConcrete::startStreaming() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (streaming_) {
//         spdlog::warn("[DataConcrete] Already streaming.");
//         return true;
//     }
//     if (!configured_ || buffers_.empty()) {
//         reportError("Device not configured or no buffers allocated.");
//         return false;
//     }

//     // Queue all buffers initially
//     for (size_t i = 0; i < buffers_.size(); ++i) {
//         if (!queueBufferInternal(i)) {
//             spdlog::error("[DataConcrete] Failed to queue buffer {}", i);
//             return false;
//         }
//     }

//     enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
//         reportError(fmt::format("VIDIOC_STREAMON failed: {}", std::strerror(errno)));
//         return false;
//     }
//     streaming_ = true;
//     spdlog::info("[DataConcrete] Streaming started with {} buffers", buffers_.size());
//     return true;
// }

// inline bool DataConcrete::stopStreaming() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (!streaming_) {
//         spdlog::warn("[DataConcrete] Not streaming; nothing to stop.");
//         return true;
//     }
//     enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     if (ioctl(fd_, VIDIOC_STREAMOFF, &type) < 0) {
//         reportError(fmt::format("VIDIOC_STREAMOFF failed: {}", std::strerror(errno)));
//     }
//     streaming_ = false;
//     spdlog::info("[DataConcrete] Streaming stopped.");
//     return true;
// }













// //====================================================================================
// bool DataConcrete::dequeFrame() {
//     struct v4l2_buffer buf;
//     memset(&buf, 0, sizeof(buf));
//     buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     buf.memory = V4L2_MEMORY_MMAP;

//     if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
//         reportError("VIDIOC_DQBUF failed");
//         return false;
//     }

//     size_t bufferIndex = buf.index;
//     Buffer& currentBuffer = buffers_[bufferIndex];

//     // Create a shared_ptr to manage the buffer's lifetime
//     auto bufferData = std::make_shared<std::vector<uint8_t>>(
//         static_cast<uint8_t*>(currentBuffer.start),
//         static_cast<uint8_t*>(currentBuffer.start) + currentBuffer.length
//     );

//     // Create ZeroCopyFrameData with the shared buffer
//     auto frame = std::make_shared<ZeroCopyFrameData>(
//         bufferData,
//         buf.bytesused,
//         cameraConfig_.width,
//         cameraConfig_.height,
//         bufferIndex,
//         framesDequeued_.fetch_add(1)
//     );

//     // Push to both queues
//     if (displayOrigQueue_) {
//         displayOrigQueue_->push(frame);
//         spdlog::debug("Zero-copy frame pushed to display queue. Queue size now: {}", displayOrigQueue_->size());
//     }

//     if (algoQueue_) {
//         algoQueue_->push(frame);
//         spdlog::debug("Zero-copy frame pushed to algorithm queue. Queue size now: {}", algoQueue_->size());
//     }

//     // Requeue the buffer immediately
//     if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
//         reportError(fmt::format("VIDIOC_QBUF error: {}", strerror(errno)));
//         return false;
//     }

//     return true;
// }

// // //====================================================================================
// // // In DataConcrete::dequeFrame
// // inline bool DataConcrete::dequeFrame(void*& dataPtr, size_t& sizeBytes, size_t& bufferIndex) {
// //     std::lock_guard<std::mutex> lock(mutex_);
// //     if (!streaming_) {
// //         reportError("[DataConcrete] Cannot dequeue frame: Not streaming.");
// //         return false;
// //     }

// //     struct v4l2_buffer buf = {};
// //     memset(&buf, 0, sizeof(buf));
// //     buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
// //     buf.memory = V4L2_MEMORY_MMAP;

// //     if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
// //         if (errno == EAGAIN) {
// //             spdlog::debug("[DataConcrete] No frame available (EAGAIN).");
// //             return false;
// //         } else {
// //             reportError(fmt::format("VIDIOC_DQBUF failed: {}", std::strerror(errno)));
// //             return false;
// //         }
// //     }
// //     if (buf.index >= buffers_.size()) {
// //         reportError(fmt::format("Dequeued buffer index out of range: {}", buf.index));
// //         return false;
// //     }
// //     spdlog::info("[DataConcrete] Successfully dequeued buffer index: {}", buf.index);

// //     auto& buffer = buffers_[buf.index];
// //     if (buffer.state != QUEUED) {
// //         reportError(fmt::format("Buffer {} is not in QUEUED state.", buf.index));
// //         return false;
// //     }

// //     bufferIndex = buf.index;
// //     buffer.state = DEQUEUED;
// //     spdlog::debug("[DataConcrete] Buffer {} dequeued.", bufferIndex);

// //     // Attempt automatic requeue: circular buffering
// //     if (!queueBuffer(bufferIndex)) {
// //         buffer.state = AVAILABLE;
// //         reportError("Failed to requeue buffer");
// //         return false;
// //     }
  
// //     Buffer& currentBuffer = buffers_[bufferIndex];

// //     // Create shared_ptr<void> with custom deleter to return buffer to the camera
// //     auto bufferData = std::shared_ptr<void>(
// //         currentBuffer.start,  // Raw pointer to buffer
// //         [this, bufferIndex](void*) {  // Custom deleter
// //             struct v4l2_buffer retBuf;
// //             memset(&retBuf, 0, sizeof(retBuf));
// //             retBuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
// //             retBuf.memory = V4L2_MEMORY_MMAP;
// //             retBuf.index = bufferIndex;
// //             ioctl(fd_, VIDIOC_QBUF, &retBuf);  // Return buffer to camera
// //         }
// //     );

// //     dataPtr   = buffer.start;
// //     sizeBytes = buf.bytesused;
// //     framesDequeued_++;

// //     spdlog::debug("[DataConcrete] Frame captured: buffer #{}, size={}, frame#={}",
// //                   bufferIndex, sizeBytes, framesDequeued_.load());

  


// //  // Initialize ZeroCopyFrameData with ALL parameters
// // //  ZeroCopyFrameData frame(
// // //     bufferData,        // bufferData
// // //     buf.bytesused,     // size
// // //     cameraConfig_.width, // width
// // //     cameraConfig_.height, // height
// // //     bufferIndex,       // bufferIndex
// // //     framesDequeued_.fetch_add(1) // frameNumber
// // // );

// // // Example for dummy initialization
// // auto dummyBuffer = std::make_shared<ZeroCopyFrameData>(
// //     std::shared_ptr<void>(), // Empty bufferData
// //     0,                      // size
// //     640,                    // width
// //     480,                    // height
// //     0,                      // bufferIndex
// //     0                       // frameNumber
// // );
// //     // Validate queue pointers
// //     if (!displayOrigQueue_ || !algoQueue_) {
// //         reportError("Queues not initialized in DataConcrete!");
// //         return false;
// //     }

// //     // Push to both queues
// //     if (displayOrigQueue_) {
// //         auto framePtr = std::make_shared<ZeroCopyFrameData>(frame);
// //         displayOrigQueue_->push(framePtr);
// //         spdlog::debug("Zero-copy frame pushed to display queue. Queue size now: {}",
// //                      displayOrigQueue_->size());
// //     }

// //     if (algoQueue_) {
// //         auto framePtr = std::make_shared<ZeroCopyFrameData>(frame);
// //         algoQueue_->push(framePtr);
// //         spdlog::debug("Zero-copy frame pushed to algorithm queue. Queue size now: {}",
// //                       algoQueue_->size());
// //     }

// //     spdlog::debug("[DataConcrete] Successfully dequeued and processed frame from buffer #{}", bufferIndex);

// //     if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
// //         reportError(fmt::format("VIDIOC_QBUF error: {}", strerror(errno)));
// //         return false;
// //     }

// //     if (static_cast<uint32_t>(cameraConfig_.pixelFormat) != V4L2_PIX_FMT_YUYV) {
// //         spdlog::warn("[DataConcrete] Camera format mismatch: expected YUYV (0x{:X})", 
// //                      static_cast<unsigned>(V4L2_PIX_FMT_YUYV));
// //     }
// //     return true;
// // }

// // // Implementation of dequeFrame using zero-copy
// // inline bool DataConcrete::dequeFrame() {
// //     std::lock_guard<std::mutex> lock(mutex_);
// //     if (!streaming_) {
// //         reportError("Not streaming; cannot dequeue frame.");
// //         return false;
// //     }

// //     v4l2_buffer buf{};
// //     buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
// //     buf.memory = V4L2_MEMORY_MMAP;

// //     if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
// //         if (errno != EAGAIN) {
// //             reportError(fmt::format("VIDIOC_DQBUF error: {}", strerror(errno)));
// //         }
// //         return false;
// //     }

// //     if (buf.index >= buffers_.size()) {
// //         reportError(fmt::format("Buffer index out of range: {}", buf.index));
// //         return false;
// //     }

// //     buffers_[buf.index].state = DEQUEUED;

// //     // ZeroCopyFrameData frame{
// //     //     buffers_[buf.index].start,
// //     //     buf.bytesused,
// //     //     cameraConfig_.width,
// //     //     cameraConfig_.height,
// //     //     buf.index
// //     // };

// //     ZeroCopyFrameData frame{
// //         .dataPtr = buffers_[buf.index].start,
// //         .size = buf.bytesused,
// //         .width = cameraConfig_.width,
// //         .height = cameraConfig_.height,
// //         .bufferIndex = buf.index,
// //         .frameNumber = framesDequeued_.load(),  // Added
// //         .captureTime = std::chrono::steady_clock::now()  // Added
// //     };

// //     if (displayOrigQueue_) {
// //         auto framePtr = std::make_shared<ZeroCopyFrameData>(frame);
// //         displayOrigQueue_->push(framePtr);
// //     }
// //     if (algoQueue_) {
// //         auto framePtr = std::make_shared<ZeroCopyFrameData>(frame);
// //         algoQueue_->push(framePtr);
// //     }

// //     buffers_[buf.index].state = QUEUED;
// //     //ioctl(fd_, VIDIOC_QBUF, &buf);

// //     if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
// //         reportError(fmt::format("VIDIOC_QBUF error: {}", strerror(errno)));
// //         return false;
// //     }

// //     framesDequeued_++;

// //     return true;
// // }

// inline bool DataConcrete::queueBuffer(size_t bufferIndex) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (bufferIndex >= buffers_.size()) {
//         reportError(fmt::format("Invalid buffer index: {}", bufferIndex));
//         return false;
//     }
//     return queueBufferInternal(bufferIndex);
// }

// inline bool DataConcrete::queueBufferInternal(size_t index) {
//     if (index >= buffers_.size()) {
//         reportError(fmt::format("queueBufferInternal: Index out of range: {}", index));
//         return false;
//     }
//     auto& buffer = buffers_[index];
//     if (buffer.state == QUEUED) {
//         reportError(fmt::format("Buffer {} is already QUEUED.", index));
//         return false;
//     }

//     struct v4l2_buffer buf = {};
//     buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     buf.memory = V4L2_MEMORY_MMAP;
//     buf.index  = index;
//     if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
//         reportError(fmt::format("VIDIOC_QBUF failed for buffer {}: {}", 
//                                 index, std::strerror(errno)));
//         return false;
//     }
//     buffer.state = QUEUED;
//     spdlog::debug("[DataConcrete] Buffer {} queued.", index);
//     return true;
// }

// inline void DataConcrete::unmapBuffers() {
//     // Make sure streaming is stopped before unmapping (done in destructor or resetDevice).
//     for (auto& buf : buffers_) {
//         if (buf.start && buf.start != MAP_FAILED) {
//             munmap(buf.start, buf.length);
//         }
//         buf.start  = nullptr;
//         buf.length = 0;
//         buf.state  = AVAILABLE;
//     }
//     buffers_.clear();
//     spdlog::debug("[DataConcrete] All buffers unmapped and cleared.");
// }

// inline void DataConcrete::reportError(const std::string& msg) {
//     if (errorCallback_) {
//         errorCallback_(msg);
//     } else {
//         spdlog::error("[DataConcrete] {}", msg);
//     }
// }

// inline void DataConcrete::setErrorCallback(std::function<void(const std::string&)> callback) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     errorCallback_ = std::move(callback);
// }

// inline bool DataConcrete::isStreaming() const {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return streaming_;
// }

// inline void DataConcrete::pushCameraMetrics() {
//     double fps = getLastFPS();
//     // For a real system, you might pass more data to PerformanceLogger
//     PerformanceLogger::getInstance().pushCameraStats(0, fps, framesQueued_.load());
// }

// inline double DataConcrete::getLastFPS() {
//     std::lock_guard<std::mutex> lock(fpsMutex_);
//     auto now      = std::chrono::steady_clock::now();
//     double elapsedSec = std::chrono::duration<double>(now - lastUpdateTime_).count();
//     if (elapsedSec > 1.0) {
//         int count = framesDequeued_.exchange(0);
//         lastFPS_  = static_cast<double>(count) / elapsedSec;
//         lastUpdateTime_ = now;
//     }
//     return lastFPS_;
// }

// inline int DataConcrete::getQueueSize() const {
//     // By default, not incremented anywhere; you can do so in `dequeFrame` or when pushing frames
//     return framesQueued_.load();
// }

// inline void DataConcrete::closeDevice() {
//     unmapBuffers();
//     if (fd_ != -1) {
//         ::close(fd_);
//         fd_ = -1;
//         spdlog::info("[DataConcrete] Device closed.");
//     }
// }

// inline void DataConcrete::resetDevice() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (streaming_) {
//         stopStreaming();
//     }
//     closeDevice();
//     spdlog::info("[DataConcrete] Device reset.");
// }

//============================================================================

// // TESTING code Final Version
///=====================TESTING  code Final Version ==============================

// // DataConcrete_new.h

#pragma once

#include "../Interfaces/IData.h"
#include "../SharedStructures/FrameData.h"
#include "../SharedStructures/SharedQueue.h"
#include "../SharedStructures/CameraConfig.h"
#include "../SharedStructures/ZeroCopyFrameData.h"
#include "../SharedStructures/ThreadManager.h" // For thread management
//#include "../../Stage_02/Logger/PerformanceLogger.h"

#include "../Interfaces/ISystemMetricsAggregator.h" // For system metrics aggregation.

#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <functional>
#include <string>
#include <spdlog/spdlog.h>

#include <linux/videodev2.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <chrono>

#include <fmt/format.h>

/**
 * @class DataConcrete
 * @brief Manages a V4L2 camera device for capturing frames and pushing them to shared queues.
 *
 * Two SharedQueue<std::shared_ptr<ZeroCopyFrameData>> are injected:
 * - algoQueue_        (frames for algorithm processing)
 * - displayOrigQueue_ (original frames for display)
 *
 * The class memory-maps a fixed number of V4L2 buffers. For each dequeued buffer,
 * the code copies the camera data into a std::vector<uint8_t> and immediately
 * re-queues the buffer. The name "ZeroCopyFrameData" is preserved for legacy,
 * but it is not truly zero-copy.
 */

class DataConcrete : public IData {
public:
    /**
     * @brief Constructor
     * @param algoQueue         Shared queue for algorithm processing
     * @param displayOrigQueue  Shared queue for display
     */
    DataConcrete(
        std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQueue,
        std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> displayOrigQueue,
        std::shared_ptr<ISystemMetricsAggregator> aggregator
    );

    /**
     * @brief Destructor: stops streaming if active, closes device
     */
    ~DataConcrete() override;

    /**
     * @brief Opens the camera device
     * @param path  The device path (e.g., "/dev/video0")
     * @return True on success, false on failure
     */
    bool openDevice(const std::string& path) override;

    /**
     * @brief Configures the camera (resolution, format, etc.)
     * @param config  Camera configuration
     * @return True on success, false otherwise
     */
    bool configure(const CameraConfig& config) override;

    /**
     * @brief Starts the camera stream (queues buffers, calls STREAMON)
     * @return True on success, false otherwise
     */
    bool startStreaming() override;

    /**
     * @brief Stops the camera stream (calls STREAMOFF)
     * @return True on success (or if already stopped)
     */
    bool stopStreaming() override;

    /**
     * @brief Starts the internal capture thread
     * @return True on success, false otherwise
     */ 
    bool startCapture() override;  // Starts internal capture threa

    void pauseCapture() {
        // Implement pause logic, e.g., set a flag
    }
    void resumeCapture() {
        // Implement resume logic
    }
    /**
     * @brief Stops the capture thread and cleans up resources
     * @return True on success, false otherwise
     */ 
    bool stopCapture() override;   // Stops thread and ensures cleanup


    /**
     * @brief Dequeues one frame, copies it to a vector, pushes it to both queues, re-queues buffer
     * @return True if successfully dequeued, false otherwise
     *
     * This function grabs a buffer from the driver via DQBUF, copies the data into a
     * std::shared_ptr<std::vector<uint8_t>>, wraps it in a ZeroCopyFrameData, and pushes
     * it to both algoQueue_ and displayOrigQueue_. It then immediately calls QBUF to return
     * the buffer to the driver.
     *
     * NOTE: If this function is called from multiple threads, concurrency must be controlled
     * to avoid conflicts with stopStreaming() or resetDevice().
     */
    bool dequeFrame() override;

    /**
     * @brief Queues a specific buffer index. Used internally, but exposed if needed externally.
     * @param bufferIndex  The buffer index
     * @return True on success, false otherwise
     */
    bool queueBuffer(size_t bufferIndex) override;

    /**
     * @brief Sets an error callback for handling error messages
     * @param callback  A function that takes a const std::string&
     */
    void setErrorCallback(std::function<void(const std::string&)>) override;

    /**
     * @brief Returns true if streaming is active
     */
    bool isStreaming() const override;

    /**
     * @brief Sends camera metrics (FPS, etc.) to the PerformanceLogger
     */
    void pushCameraMetrics() override;

    /**
     * @brief Computes or returns the most recently computed FPS
     * @return Current frames-per-second estimate
     */
    double getLastFPS() override;

    /**
     * @brief Returns the number of frames queued (if you update framesQueued_)
     */
    int getQueueSize() const override;

    /**
     * @brief Closes the device (unmaps buffers, resets file descriptor)
     */
    void closeDevice();

    /**
     * @brief Stops streaming (if active), then closes and resets the device
     */
    void resetDevice();

    /**
     * @brief Getters for the two shared queues
     */
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> getDisplayQueue() const {
        return displayOrigQueue_;
    }
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> getProcessingQueue() const {
        return algoQueue_;
    }

private:
    /**
     * @enum BufferState
     * @brief Possible buffer states (some are unused in this example)
     */
    enum BufferState {
        AVAILABLE,
        QUEUED,
        DEQUEUED,
        PROCESSING,
        IN_USE
    };

    /**
     * @struct Buffer
     * @brief Manages a memory-mapped buffer's pointer, length, and state
     */
    struct Buffer {
        void*       start = nullptr;
        size_t      length = 0;
        BufferState state  = AVAILABLE;
        std::chrono::steady_clock::time_point captureTime;
    };

    /**
     * @brief Initializes memory-mapped buffers by calling QUERYBUF + mmap
     * @return True on success, false otherwise
     */
    bool initializeBuffers();

    /**
     * @brief Internal helper for queueing a buffer
     * @param index  The buffer index
     * @return True on success, false otherwise
     */
    bool queueBufferInternal(size_t index);

    /**
     * @brief Unmaps all buffers and clears them
     */
    void unmapBuffers();

    /**
     * @brief Logs or reports an error
     * @param msg  The error message
     */
    void reportError(const std::string& msg);

    /**
     * @brief Placeholder for additional cleanup if needed
     */
    void cleanUpResources() {}

    // Device file descriptor
    int fd_ = -1;

    // Streaming state
    std::atomic<bool> streaming_{false};

    // Configuration state
    bool configured_ = false;

    // Device path (e.g., "/dev/video0")
    std::string devicePath_;

    // Camera configuration parameters
    CameraConfig cameraConfig_;

    // Memory-mapped buffers
    std::vector<Buffer> buffers_;

    // Lock for device-wide operations (open, config, start/stop, etc.)
    mutable std::mutex mutex_;

    // Optional lock for fine-grained buffer management if needed
    mutable std::mutex bufferMutex_;

    // Number of buffers to allocate
    static constexpr int NUM_BUFFERS = 4;

    // Optional error callback
    std::function<void(const std::string&)> errorCallback_;

    // FPS tracking
    mutable std::mutex fpsMutex_;
    mutable double lastFPS_ = 0.0;
    mutable std::atomic<int> framesDequeued_{0};
    mutable std::chrono::steady_clock::time_point lastUpdateTime_;

    // Total frames queued count (not actively updated in this code)
    std::atomic<int> framesQueued_{0};

    // Shared queues for the pipeline
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQueue_;
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> displayOrigQueue_;

    // DataConcrete to support SystemMetricsAggregatorImpl injection and push camera metrics in real-time using the aggregator
    std::shared_ptr<ISystemMetricsAggregator> metricAggregator_;

    /**
     * @brief Thread function for capturing frames, Thread Management Internally
     * 
     */
    void captureThreadFunc();  // Internal thread function

    std::atomic<bool> running_{false};
    std::thread captureThread_;
    std::atomic<uint64_t> frameCounter_{0};

};

// ------------------ Implementation ------------------

inline DataConcrete::DataConcrete(
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQueue,
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> displayOrigQueue,
    std::shared_ptr<ISystemMetricsAggregator> aggregator
)
  : fd_(-1)
  , streaming_(false)
  , configured_(false)
  , lastFPS_(0.0)
  , algoQueue_(std::move(algoQueue))
  , displayOrigQueue_(std::move(displayOrigQueue))
  , metricAggregator_(std::move(aggregator))
{
    lastUpdateTime_ = std::chrono::steady_clock::now();
    spdlog::debug("[DataConcrete] Constructor complete.");
}

inline DataConcrete::~DataConcrete() {
    if (running_) {
        running_ = false;
        stopCapture();  // Ensures thread stops

    }   

    if (streaming_) {
        stopStreaming();
    }
    if (fd_ >= 0) {
        closeDevice();
    }

    stopStreaming();
    closeDevice();

    spdlog::debug("[DataConcrete] Destructor complete.");
}

inline bool DataConcrete::openDevice(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fd_ != -1) {
        spdlog::warn("[DataConcrete] Device already open. Closing previous device.");
        closeDevice();
    }
    fd_ = ::open(path.c_str(), O_RDWR);
    if (fd_ < 0) {
        reportError(fmt::format("Failed to open {}: {}", path, std::strerror(errno)));
        return false;
    }
    devicePath_ = path;
    spdlog::info("[DataConcrete] Device opened: {}", path);
    return true;
}

inline bool DataConcrete::configure(const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fd_ < 0) {
        reportError("Device not opened; cannot configure.");
        return false;
    }
    // If already configured, unmap old buffers
    if (configured_ || !buffers_.empty()) {
        spdlog::info("[DataConcrete] Re-configuring device. Unmapping old buffers.");
        unmapBuffers();
    }

    cameraConfig_ = config;

    // Set format (YUYV in this example)
    struct v4l2_format fmt = {};
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = config.width;
    fmt.fmt.pix.height      = config.height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        reportError(fmt::format("VIDIOC_S_FMT failed: {}", std::strerror(errno)));
        return false;
    }

    // Validate the set format
    struct v4l2_format currentFmt = {};
    currentFmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_G_FMT, &currentFmt) < 0) {
        reportError(fmt::format("VIDIOC_G_FMT failed after S_FMT: {}", std::strerror(errno)));
        return false;
    }
    if (currentFmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
        reportError(fmt::format(
            "Format mismatch after S_FMT: Expected YUYV (0x{:X}), got 0x{:X}",
            static_cast<unsigned>(V4L2_PIX_FMT_YUYV),
            static_cast<unsigned>(currentFmt.fmt.pix.pixelformat)
        ));
        return false;
    } else {
        spdlog::info("[DataConcrete] Confirmed camera format is YUYV (0x{:X}).",
                     static_cast<unsigned>(currentFmt.fmt.pix.pixelformat));
    }

    // Request buffers
    struct v4l2_requestbuffers req = {};
    req.count  = NUM_BUFFERS;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        reportError(fmt::format("VIDIOC_REQBUFS failed: {}", std::strerror(errno)));
        return false;
    }

    // Initialize (mmap) buffers
    if (!initializeBuffers()) {
        return false;
    }
    configured_ = true;
    spdlog::info("[DataConcrete] Device configured ({}x{}).", config.width, config.height);
    return true;
}

inline bool DataConcrete::initializeBuffers() {
    buffers_.clear();
    buffers_.resize(NUM_BUFFERS);

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        struct v4l2_buffer buf = {};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;

        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            reportError(fmt::format("VIDIOC_QUERYBUF failed for buffer {}: {}",
                                    i, std::strerror(errno)));
            return false;
        }

        buffers_[i].length = buf.length;
        buffers_[i].start  = mmap(nullptr, buf.length,
                                  PROT_READ | PROT_WRITE,
                                  MAP_SHARED,
                                  fd_,
                                  buf.m.offset);
        if (buffers_[i].start == MAP_FAILED) {
            reportError(fmt::format("mmap failed for buffer {}: {}", i, std::strerror(errno)));
            return false;
        }
        buffers_[i].state = AVAILABLE;
        spdlog::debug("[DataConcrete] Buffer {} mapped, size: {}", i, buf.length);
    }
    return true;
}

inline bool DataConcrete::startStreaming() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (streaming_) {
        spdlog::warn("[DataConcrete] Already streaming.");
        return true;
    }
    if (!configured_ || buffers_.empty()) {
        reportError("Device not configured or no buffers allocated.");
        return false;
    }

    // Queue all buffers initially
    for (size_t i = 0; i < buffers_.size(); ++i) {
        if (!queueBufferInternal(i)) {
            spdlog::error("[DataConcrete] Failed to queue buffer {}", i);
            return false;
        }
    }

    // Turn on streaming
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        reportError(fmt::format("VIDIOC_STREAMON failed: {}", std::strerror(errno)));
        return false;
    }

    streaming_ = true;
    spdlog::info("[DataConcrete] Streaming started with {} buffers.", buffers_.size());
    return true;
}

inline bool DataConcrete::stopStreaming() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!streaming_) {
        spdlog::warn("[DataConcrete] Not streaming; nothing to stop.");
        return true;
    }
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMOFF, &type) < 0) {
        reportError(fmt::format("VIDIOC_STREAMOFF failed: {}", std::strerror(errno)));
    }
    streaming_ = false;
    spdlog::info("[DataConcrete] Streaming stopped.");
    return true;
}


/**
 * @brief   Starts the capture thread
 * 
 * @return true 
 * @return false 
 */
inline bool DataConcrete::startCapture() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) return true;  // Already running
    
    if (streaming_) {
        spdlog::warn("[DataConcrete] Capture already started.");
        return true;
    }
    if (!startStreaming()) {
        return false;  // Failed to start camera stream
    }

    if (!configured_) {
        reportError("Device not configured; cannot start capture.");
        return false;
    }
    running_ = true;
    streaming_ = true;

    captureThread_ = std::thread(&DataConcrete::captureThreadFunc, this);
    

    spdlog::info("[DataConcrete] Camera Capture thread started.");
    return true;
}


// This function is called to stop the capture thread
/**
 * @brief   Stops the capture thread
 * 
 * @return true 
 * @return false 
 */
inline bool DataConcrete::stopCapture() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) return true;

    running_ = false;
    
    if (captureThread_.joinable()) {
        captureThread_.join();
    }

    if (!streaming_) {
        spdlog::warn("[DataConcrete] Capture already stopped.");
        return true;
    }
    stopStreaming();  // Ensure V4L2 stream is stopped

    streaming_ = false;
    //stopStreaming();  // Ensure V4L2 stream is stopped
    spdlog::info("[DataConcrete] Camera capture thread stopped.");

 if(fd_ >= 0)
    {
        for(auto& b: buffers_)
            if(b.start && b.length) { munmap(b.start, b.length); }

        ::close(fd_);
        fd_ = -1;
        //spdlog::info(\"[DataConcrete] /dev/video device closed.\");
    }

    return true;
}

/**
 * @brief   Capture thread function
 * 
 * This function runs in a separate thread and continuously dequeues frames from the camera.
 * It pushes the frames to the algoQueue_ and displayOrigQueue_.
 */
void DataConcrete::captureThreadFunc() {
    while (running_) {
        if (!dequeFrame()) { // Automatically pushes metrics
            if (errno == EAGAIN) {
                std::this_thread::yield();  // No frame ready
                continue;
            }
            reportError("Failed to dequeue frame");
            break;
        }

        // Metrics are automatically pushed in `dequeFrame()`
    }
}


/**
 * @brief Dequeues a frame, copies it to a vector, pushes it to both queues, re-queues buffer
 * @return True if successfully dequeued, false otherwise
 *
 * This function grabs a buffer from the driver via DQBUF, copies the data into a
 * std::shared_ptr<std::vector<uint8_t>>, wraps it in a ZeroCopyFrameData, and pushes
 * it to both algoQueue_ and displayOrigQueue_. It then immediately calls QBUF to return
 * the buffer to the driver.
 *
 * NOTE: If this function is called from multiple threads, concurrency must be controlled
 * to avoid conflicts with stopStreaming() or resetDevice().
 */
inline bool DataConcrete::dequeFrame() {
    // Lock here to prevent race conditions with resetDevice() or stopStreaming()
    std::lock_guard<std::mutex> lock(mutex_);

    if (!streaming_) {
        spdlog::warn("[DataConcrete] dequeFrame called but not streaming.");
        return false;
    }

    // Dequeue a buffer
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        reportError(fmt::format("VIDIOC_DQBUF failed: {}", std::strerror(errno)));
        return false;
    }

    // Copy data from mmap buffer
    size_t bufferIndex = buf.index;
    Buffer& currentBuffer = buffers_[bufferIndex];

    auto bufferData = std::make_shared<std::vector<uint8_t>>(
        static_cast<uint8_t*>(currentBuffer.start),
        static_cast<uint8_t*>(currentBuffer.start) + currentBuffer.length
    );

    // Build the ZeroCopyFrameData (despite the copy, name is historical)
    // auto frame = std::make_shared<ZeroCopyFrameData>(
    //     bufferData,
    //     buf.bytesused,
    //     cameraConfig_.width,
    //     cameraConfig_.height,
    //     bufferIndex,
    //     framesDequeued_.fetch_add(1) // increment & return old value
    // );

    // In DataConcrete::dequeFrame
auto frame = std::make_shared<ZeroCopyFrameData>(
    bufferData,          // Pass the shared_ptr to the vector
    bufferData->size(),  // Correct size
    cameraConfig_.width,
    cameraConfig_.height,
    bufferIndex,
    framesDequeued_.fetch_add(1)
);

    // Push to display queue
    if (displayOrigQueue_) {
        displayOrigQueue_->push(frame);
        spdlog::debug("[DataConcrete] Pushed frame to display queue. Size now: {}",
                      displayOrigQueue_->size());
    }

    // Push to algorithm queue
    if (algoQueue_) {
        algoQueue_->push(frame);
        spdlog::debug("[DataConcrete] Pushed frame to algo queue. Size now: {}",
                      algoQueue_->size());
    }
    //Metric Push
    /**
     * @brief Construct a new if object
     * Inside DataConcrete::dequeFrame():
     * Metrics are pushed automatically for every dequeued frame
     * Includes comprehensive camera stats (timestamp, frame number, FPS, resolution, frame size)
     * Uses the injected metric aggregator directly
     */
    if (metricAggregator_) {
        auto now = std::chrono::system_clock::now();
        double fps = getLastFPS();
        
        // metricAggregator_->pushMetrics(now, [this, now](SystemMetricsSnapshot& snapshot) {
        //     snapshot.timestamp = now;
        metricAggregator_->pushMetrics(now, [this, now, fps](SystemMetricsSnapshot& snapshot) {
        snapshot.timestamp = now;
    
        // Camera stats
        snapshot.cameraStats.timestamp   = now;
        snapshot.cameraStats.frameNumber = framesDequeued_.load();
        snapshot.cameraStats.fps         = getLastFPS();
        snapshot.cameraStats.frameWidth  = cameraConfig_.width;
        snapshot.cameraStats.frameHeight = cameraConfig_.height;
        snapshot.cameraStats.frameSize   = cameraConfig_.width * cameraConfig_.height * 2; // assuming YUYV
        });
    }


    // inside DataConcrete::dequeFrame(), after frame captured:
    
        CameraStats cs;
        cs.timestamp   = std::chrono::system_clock::now();
        cs.frameNumber = frameCounter_++;
        cs.fps         = lastFPS_;
        cs.frameWidth  = cameraConfig_.width;
        cs.frameHeight = cameraConfig_.height;
        metricAggregator_->beginFrame(cs.frameNumber, cs);
    
    
    


    // Re-queue buffer
    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        reportError(fmt::format("VIDIOC_QBUF failed: {}", std::strerror(errno)));
        return false;
    }

    return true;
}

inline bool DataConcrete::queueBuffer(size_t bufferIndex) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bufferIndex >= buffers_.size()) {
        reportError(fmt::format("Invalid buffer index: {}", bufferIndex));
        return false;
    }
    return queueBufferInternal(bufferIndex);
}

inline bool DataConcrete::queueBufferInternal(size_t index) {
    if (index >= buffers_.size()) {
        reportError(fmt::format("queueBufferInternal: Index out of range: {}", index));
        return false;
    }

    auto& buffer = buffers_[index];
    if (buffer.state == QUEUED) {
        reportError(fmt::format("Buffer {} is already QUEUED.", index));
        return false;
    }

    struct v4l2_buffer buf = {};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index  = index;

    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        reportError(fmt::format("VIDIOC_QBUF failed for buffer {}: {}",
                                index, std::strerror(errno)));
        return false;
    }
    buffer.state = QUEUED;
    spdlog::debug("[DataConcrete] Buffer {} queued.", index);
    return true;
}

inline void DataConcrete::unmapBuffers() {
    for (auto& buf : buffers_) {
        if (buf.start && buf.start != MAP_FAILED) {
            munmap(buf.start, buf.length);
        }
        buf.start  = nullptr;
        buf.length = 0;
        buf.state  = AVAILABLE;
    }
    buffers_.clear();
    spdlog::debug("[DataConcrete] All buffers unmapped and cleared.");
}

inline void DataConcrete::reportError(const std::string& msg) {
    if (errorCallback_) {
        errorCallback_(msg);
    } else {
        spdlog::error("[DataConcrete] {}", msg);
    }
}

inline void DataConcrete::setErrorCallback(std::function<void(const std::string&)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    errorCallback_ = std::move(callback);
}

inline bool DataConcrete::isStreaming() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return streaming_;
}

// pushCameraMetrics NO used!!
inline void DataConcrete::pushCameraMetrics() {
   // double fps = getLastFPS();
    // framesQueued_ is never updated in this example, so it remains zero
   // PerformanceLogger::getInstance().pushCameraStats(0, fps, framesQueued_.load());
}

inline double DataConcrete::getLastFPS() {
    std::lock_guard<std::mutex> lock(fpsMutex_);
    auto now        = std::chrono::steady_clock::now();
    double elapsed  = std::chrono::duration<double>(now - lastUpdateTime_).count();

    if (elapsed > 1.0) {
        int count = framesDequeued_.exchange(0);
        lastFPS_  = static_cast<double>(count) / elapsed;
        lastUpdateTime_ = now;
    }

    return lastFPS_;
}

inline int DataConcrete::getQueueSize() const {
    // Not actively updated; you could increment framesQueued_ whenever pushing frames
    return framesQueued_.load();
}

inline void DataConcrete::closeDevice() {
    unmapBuffers();
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
        spdlog::info("[DataConcrete] Device closed.");
    }
}

inline void DataConcrete::resetDevice() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (streaming_) {
        stopStreaming();
    }
    closeDevice();
    spdlog::info("[DataConcrete] Device reset.");
}
//====================================================================================

//=================================   13-05-2025 Version ===================================================
// // // DataConcrete_new.h

// DataConcrete_v2.h  ? synchronised?frame version
// -----------------------------------------------------------------------------
//  * Calls metricAggregator_->beginFrame(frameId, cameraStats) ONCE per frame.
//  * Removed legacy pushMetrics() block (no duplicates).
//  * Guarantees camera fd is closed & buffers unmapped on shutdown.
//  * Adds frameCounter_ (atomic) -> authoritative frameId for pipeline.
// //  * Provides startCapture/stopCapture helpers for threaded capture.
// // -----------------------------------------------------------------------------
// #pragma once



// #include "../Interfaces/IData.h"
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/SharedQueue.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/ZeroCopyFrameData.h"
// #include "../SharedStructures/ThreadManager.h" // For thread management
// //#include "../../Stage_02/Logger/PerformanceLogger.h"

// #include "../Interfaces/ISystemMetricsAggregator.h" // For system metrics aggregation.
// #include "../SharedStructures/allModulesStatcs.h"

// #include <vector>
// #include <atomic>
// #include <mutex>
// #include <thread>
// #include <functional>
// #include <string>
// #include <spdlog/spdlog.h>

// #include <linux/videodev2.h>
// #include <sys/mman.h>
// #include <fcntl.h>
// #include <sys/ioctl.h>
// #include <unistd.h>
// #include <cerrno>
// #include <cstring>
// #include <chrono>

// #include <fmt/format.h>


// // #include "../Interfaces/IData.h"
// // #include "../SharedStructures/ZeroCopyFrameData.h"
// // #include "../SharedStructures/SharedQueue.h"
// // #include "../SharedStructures/CameraConfig.h"
// // #include "../Interfaces/ISystemMetricsAggregator.h"
// // #include "../SharedStructures/ThreadManager.h"

// // #include <atomic>
// // #include <chrono>
// // #include <fcntl.h>
// // #include <linux/videodev2.h>
// // #include <mutex>
// // #include <spdlog/spdlog.h>
// // #include <string>
// // #include <sys/ioctl.h>
// // #include <sys/mman.h>
// // #include <unistd.h>
// // #include <vector>

// class DataConcrete : public IData
// {
// public:
//     DataConcrete(std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQ,
//                  std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> dispQ,
//                  std::shared_ptr<ISystemMetricsAggregator> agg);
//     ~DataConcrete() override;

//     bool openDevice (const std::string& path) override;
//     bool configure  (const CameraConfig&)   override;
//     bool startStreaming()                   override;
//     bool stopStreaming ()                   override;

//     bool startCapture() override;   // spawns capture thread
//     bool stopCapture () override;

//    // Additional device management methods
//     void closeDevice();
//     void resetDevice();


//     bool dequeFrame() override;     // used by optional external caller (not used by thread)
//     bool queueBuffer(size_t) override;

//     void setErrorCallback(std::function<void(const std::string&)>) override;
//     bool isStreaming() const override { return streaming_; }
//     void pushCameraMetrics() override {}
//     double getLastFPS() override;
//     int    getQueueSize() const override { return 0; }

// private:
//     enum class BufState { FREE, QUEUED, IN_FLIGHT };
//     struct Buffer { void* start=nullptr; size_t len=0; BufState st=BufState::FREE; };

//     bool  initializeBuffers();
//     bool  queueBufferInternal(size_t);
//     void  unmapBuffers();
//     void  reportError(const std::string&);
//     void  captureLoop();

//     //------------------------------------------------------------------
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQ_;
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> dispQ_;
//     std::shared_ptr<ISystemMetricsAggregator> agg_;

//     int fd_ = -1;
//     CameraConfig cfg_{};

//     std::vector<Buffer> bufs_;
//     static constexpr int NUM_BUF = 4;

//     std::atomic<bool> streaming_{false};
//     std::atomic<bool> running_{false};
//     std::thread       capThread_;

//     // FPS
//     std::atomic<uint64_t> frameCounter_{0};
//     std::atomic<int>      fpsCount_{0};
//     std::chrono::steady_clock::time_point fpsT0_ = std::chrono::steady_clock::now();
//     double lastFps_=0.0;

//     // sync
//     std::mutex m_;
//     std::function<void(const std::string&)> errCb_;
//     // Lock for device-wide operations (open, config, start/stop, etc.)
//      mutable std::mutex mutex_;
// };

// // ===========================================================================
// inline DataConcrete::DataConcrete(std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> a,
//                                   std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> d,
//                                   std::shared_ptr<ISystemMetricsAggregator> agg)
//     : algoQ_(std::move(a)), dispQ_(std::move(d)), agg_(std::move(agg))
// {
//     spdlog::debug("[DataConcrete] ctor");
// }

// inline DataConcrete::~DataConcrete()
// {
//     stopCapture();
//     stopStreaming();
//     unmapBuffers();
//     if(fd_!=-1) ::close(fd_);
// }

// // ---------------- V4L2 helpers ---------------------------------------------
// inline bool DataConcrete::openDevice(const std::string& p)
// {
//     std::lock_guard<std::mutex> lk(m_);
//     if(fd_!=-1) { ::close(fd_); fd_=-1; }
//     //fd_ = ::open(p.c_str(),O_RDWR);
//     //if(fd_<0) return reportError("open("+p+") failed"), false;


//     fd_=::open(p.c_str(), O_RDWR|O_NONBLOCK|O_CLOEXEC);
//     if(fd_<0){ reportError("open("+p+") "+strerror(errno)); return false; }

//     spdlog::info("[DataConcrete] opened {}",p);
//     return true;
// }

// inline bool DataConcrete::configure(const CameraConfig& c)
// {
//     std::lock_guard<std::mutex> lk(m_);
//     cfg_=c;
//     // set format
//     v4l2_format fmt{}; fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     fmt.fmt.pix.width=c.width; fmt.fmt.pix.height=c.height;
//     fmt.fmt.pix.pixelformat=V4L2_PIX_FMT_YUYV; fmt.fmt.pix.field=V4L2_FIELD_NONE;


//     if(ioctl(fd_,VIDIOC_S_FMT,&fmt)<0) return reportError("S_FMT"),false;

//     // 
//     auto trySetFmt=[&](){ return ioctl(fd_,VIDIOC_S_FMT,&fmt)==0; };
//     if(!trySetFmt()){
//         if(errno!=EBUSY){ reportError("VIDIOC_S_FMT: "+std::string(strerror(errno))); return false; }
//         // EBUSY recovery path ---------------------------------------------
//         spdlog::warn("[DataConcrete] Device busy ? attempting forced reset.");
//         v4l2_buf_type t=V4L2_BUF_TYPE_VIDEO_CAPTURE; ioctl(fd_,VIDIOC_STREAMOFF,&t);
//         v4l2_requestbuffers req{}; req.count=0; req.type=t; req.memory=V4L2_MEMORY_MMAP; ioctl(fd_,VIDIOC_REQBUFS,&req);
//         usleep(100000);  // 100?ms breathe
//         if(!trySetFmt()){
//             reportError("VIDIOC_S_FMT still busy: "+std::string(strerror(errno))); return false;
//         }
//     }

//     // request bufs
//     v4l2_requestbuffers req{}; req.count=NUM_BUF; req.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; req.memory=V4L2_MEMORY_MMAP;
//     if(ioctl(fd_,VIDIOC_REQBUFS,&req)<0) return reportError("REQBUFS"),false;

//     return initializeBuffers();
// }

// inline bool DataConcrete::initializeBuffers()
// {
//     bufs_.resize(NUM_BUF);
//     for(int i=0;i<NUM_BUF;++i){
//         v4l2_buffer b{}; b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory=V4L2_MEMORY_MMAP; b.index=i;
//         if(ioctl(fd_,VIDIOC_QUERYBUF,&b)<0) return reportError("QUERYBUF"),false;
//         void* p=mmap(nullptr,b.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,b.m.offset);
//         if(p==MAP_FAILED) return reportError("mmap"),false;
//         bufs_[i]={p,b.length,BufState::FREE};
//     }
//     return true;
// }

// inline bool DataConcrete::queueBufferInternal(size_t idx)
// {
//     v4l2_buffer b{}; b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory=V4L2_MEMORY_MMAP; b.index=idx;
//     if(ioctl(fd_,VIDIOC_QBUF,&b)<0) return reportError("QBUF"),false;
//     bufs_[idx].st = BufState::QUEUED; return true;
// }

// //inline bool DataConcrete::startStreaming(){ for(int i=0;i<NUM_BUF;++i) if(!queueBufferInternal(i)) return false; v4l2_buf_type t=V4L2_BUF_TYPE_VIDEO_CAPTURE; if(ioctl(fd_,VIDIOC_STREAMON,&t)<0) return reportError("STREAMON"),false; streaming_=true; return true; }
// //inline bool DataConcrete::stopStreaming (){ if(!streaming_) return true; v4l2_buf_type t=V4L2_BUF_TYPE_VIDEO_CAPTURE; ioctl(fd_,VIDIOC_STREAMOFF,&t); streaming_=false; return true; }


// inline bool DataConcrete::startStreaming()
// {
//     for(size_t i=0;i<NUM_BUF;++i) if(!queueBufferInternal(i)) return false;
//     v4l2_buf_type t=V4L2_BUF_TYPE_VIDEO_CAPTURE; if(ioctl(fd_,VIDIOC_STREAMON,&t)<0) return reportError("STREAMON"),false;
//     streaming_=true; return true;
// }

// inline bool DataConcrete::stopStreaming()
// {
//     if(!streaming_) return true;
//     v4l2_buf_type t=V4L2_BUF_TYPE_VIDEO_CAPTURE; ioctl(fd_,VIDIOC_STREAMOFF,&t);
//     streaming_=false; return true;
// }

// // ---------------- capture thread -------------------------------------------
// //inline bool DataConcrete::startCapture(){ if(running_) return true; if(!startStreaming()) return false; running_=true; capThread_=std::thread(&DataConcrete::captureLoop,this); return true; }
// //inline bool DataConcrete::stopCapture (){ running_=false; if(capThread_.joinable()) capThread_.join(); stopStreaming(); return true; }

// // closeDevice() and resetDevice() are called from the destructor
// inline void DataConcrete::closeDevice() {
//     unmapBuffers();
//     if (fd_ != -1) {
//         ::close(fd_);
//         fd_ = -1;
//         spdlog::info("[DataConcrete] Device closed.");
//     }
// }

// inline void DataConcrete::resetDevice() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (streaming_) {
//         stopStreaming();
//     }
//     closeDevice();
//     spdlog::info("[DataConcrete] Device reset.");
// }

// // ---------------- Capture thread -------------------------------------------
// inline bool DataConcrete::startCapture()
// {
//     if(running_) return true;
//     if(!startStreaming()) return false;
//     running_=true; capThread_=std::thread(&DataConcrete::captureLoop,this);
//     return true;
// }

// inline bool DataConcrete::stopCapture()
// {
//     running_=false;
//     if(capThread_.joinable()) capThread_.join();
//     stopStreaming();
//     return true;
// }

// inline void DataConcrete::captureLoop()
// {
//     while(running_){
//         if(!dequeFrame()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
//     }
// }


// //inline void DataConcrete::captureLoop(){ while(running_){ if(!dequeFrame()) std::this_thread::sleep_for(std::chrono::milliseconds(5)); } }



// inline bool DataConcrete::dequeFrame()
// {
//     if(!streaming_) return false;

//     v4l2_buffer b{}; b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory=V4L2_MEMORY_MMAP;
//     if(ioctl(fd_,VIDIOC_DQBUF,&b)<0) return false;

//     auto& bf = bufs_[b.index];
//     const size_t bytes = b.bytesused;

//     // copy -> vector (keep simple for now)
//     auto vec = std::make_shared<std::vector<uint8_t>>(static_cast<uint8_t*>(bf.start),static_cast<uint8_t*>(bf.start)+bytes);
//     uint64_t fid = frameCounter_.fetch_add(1);

//     auto zf = std::make_shared<ZeroCopyFrameData>(vec,bytes,cfg_.width,cfg_.height,b.index,fid);
//     if(dispQ_)  dispQ_->push(zf);
//     if(algoQ_)  algoQ_->push(zf);

//     // FPS update
//     fpsCount_++;
//     auto now = std::chrono::steady_clock::now();
//     double sec = std::chrono::duration<double>(now-fpsT0_).count();
//     if(sec>=1.0){ lastFps_=fpsCount_/sec; fpsCount_=0; fpsT0_=now; }

//     if(agg_){
//         CameraStats cs; cs.timestamp=std::chrono::system_clock::now(); cs.frameNumber=fid;
//         cs.fps=lastFps_; cs.frameWidth=cfg_.width; cs.frameHeight=cfg_.height; cs.frameSize=bytes;
//         agg_->beginFrame(fid,cs);
//     }

//     // re?queue
//     queueBufferInternal(b.index);
//     return true;
// }

// // ----------------------------------------------------------------------------
// inline double DataConcrete::getLastFPS(){ return lastFps_; }

// inline void DataConcrete::unmapBuffers()
// {
//     for(auto& b: bufs_) if(b.start && b.start!=MAP_FAILED) munmap(b.start,b.len);
//     bufs_.clear();
// }

// inline void DataConcrete::reportError(const std::string& m)
// {
//     if(errCb_) errCb_(m); else spdlog::error("[DataConcrete] {}",m);
// }

// inline bool DataConcrete::queueBuffer(size_t idx){ return queueBufferInternal(idx); }

// inline void DataConcrete::setErrorCallback(std::function<void(const std::string&)> cb){ errCb_=std::move(cb);}


//=================================   23-05-2025 Version ===================================================


// #pragma once

// #include "../Interfaces/IData.h"
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/SharedQueue.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/ZeroCopyFrameData.h"
// #include "../SharedStructures/ThreadManager.h"
// #include "../Interfaces/ISystemMetricsAggregator.h"
// #include "../SharedStructures/allModulesStatcs.h"

// #include <vector>
// #include <atomic>
// #include <mutex>
// #include <thread>
// #include <functional>
// #include <string>
// #include <spdlog/spdlog.h>
// #include <linux/videodev2.h>
// #include <sys/mman.h>
// #include <fcntl.h>
// #include <sys/ioctl.h>
// #include <unistd.h>
// #include <cerrno>
// #include <cstring>
// #include <chrono>
// #include <fmt/format.h>

// class BufferManager {
// public:
//     using RequeueCallback = std::function<void(size_t)>;
//     BufferManager(RequeueCallback callback) : requeueCallback_(std::move(callback)) {}
    
//     std::shared_ptr<void> acquireBuffer(size_t bufferIndex, void* data) {
//         std::lock_guard<std::mutex> lock(mutex_);
//         if (bufferIndex >= refCounts_.size()) refCounts_.resize(bufferIndex + 1, 0);
//         refCounts_[bufferIndex]++;
//         return std::shared_ptr<void>(data, [this, bufferIndex](void*) {
//             releaseBuffer(bufferIndex);
//         });
//     }

// private:
//     void releaseBuffer(size_t bufferIndex) {
//         std::lock_guard<std::mutex> lock(mutex_);
//         if (bufferIndex >= refCounts_.size() || refCounts_[bufferIndex] == 0) return;
//         if (--refCounts_[bufferIndex] == 0) requeueCallback_(bufferIndex);
//     }

//     std::vector<int> refCounts_;
//     std::mutex mutex_;
//     RequeueCallback requeueCallback_;
// };

// class DataConcrete : public IData {
// public:
//     DataConcrete(std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQ,
//                  std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> dispQ,
//                  std::shared_ptr<ISystemMetricsAggregator> agg)
//         : algoQ_(std::move(algoQ)), dispQ_(std::move(dispQ)), agg_(std::move(agg)),
//           bufferManager_([this](size_t idx) { queueBufferInternal(idx); }) {
//         spdlog::debug("[DataConcrete] ctor");
//     }

//     ~DataConcrete() override {
//         stopCapture();
//         stopStreaming();
//         unmapBuffers();
//         if (fd_ != -1) ::close(fd_);
//     }

//     bool openDevice(const std::string& p) override {
//         std::lock_guard<std::mutex> lk(m_);
//         if (fd_ != -1) { ::close(fd_); fd_ = -1; }
//         fd_ = ::open(p.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
//         if (fd_ < 0) { reportError("open(" + p + ") " + strerror(errno)); return false; }
//         spdlog::info("[DataConcrete] opened {}", p);
//         return true;
//     }

//     bool configure(const CameraConfig& c) override {
//         std::lock_guard<std::mutex> lk(m_);
//         cfg_ = c;
//         v4l2_format fmt{}; fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//         fmt.fmt.pix.width = c.width; fmt.fmt.pix.height = c.height;
//         fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV; fmt.fmt.pix.field = V4L2_FIELD_NONE;
//         if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
//             if (errno != EBUSY) { reportError("VIDIOC_S_FMT: " + std::string(strerror(errno))); return false; }
//             spdlog::warn("[DataConcrete] Device busy, attempting reset.");
//             v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE; ioctl(fd_, VIDIOC_STREAMOFF, &t);
//             v4l2_requestbuffers req{}; req.count = 0; req.type = t; req.memory = V4L2_MEMORY_MMAP; ioctl(fd_, VIDIOC_REQBUFS, &req);
//             usleep(100000);
//             if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) { reportError("VIDIOC_S_FMT still busy: " + std::string(strerror(errno))); return false; }
//         }
//         v4l2_requestbuffers req{}; req.count = NUM_BUF; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; req.memory = V4L2_MEMORY_MMAP;
//         if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) return reportError("REQBUFS"), false;
//         return initializeBuffers();
//     }

//     bool initializeBuffers() {
//         bufs_.resize(NUM_BUF);
//         for (int i = 0; i < NUM_BUF; ++i) {
//             v4l2_buffer b{}; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; b.index = i;
//             if (ioctl(fd_, VIDIOC_QUERYBUF, &b) < 0) return reportError("QUERYBUF"), false;
//             void* p = mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, b.m.offset);
//             if (p == MAP_FAILED) return reportError("mmap"), false;
//             bufs_[i] = {p, b.length, BufState::FREE};
//         }
//         return true;
//     }

//     bool startStreaming() override {
//         for (size_t i = 0; i < NUM_BUF; ++i) if (!queueBufferInternal(i)) return false;
//         v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE; if (ioctl(fd_, VIDIOC_STREAMON, &t) < 0) return reportError("STREAMON"), false;
//         streaming_ = true; return true;
//     }

//     bool stopStreaming() override {
//         if (!streaming_) return true;
//         v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE; ioctl(fd_, VIDIOC_STREAMOFF, &t);
//         streaming_ = false; return true;
//     }

//     bool startCapture() override {
//         if (running_) return true;
//         if (!startStreaming()) return false;
//         running_ = true; capThread_ = std::thread(&DataConcrete::captureLoop, this);
//         return true;
//     }

//     bool stopCapture() override {
//         running_ = false;
//         if (capThread_.joinable()) capThread_.join();
//         stopStreaming();
//         return true;
//     }

//     void closeDevice() {
//         unmapBuffers();
//         if (fd_ != -1) {
//             ::close(fd_);
//             fd_ = -1;
//             spdlog::info("[DataConcrete] Device closed.");
//         }
//     }

//     void resetDevice() {
//         std::lock_guard<std::mutex> lock(m_);
//         if (streaming_) stopStreaming();
//         closeDevice();
//         spdlog::info("[DataConcrete] Device reset.");
//     }

//     void captureLoop() {
//         while (running_) {
//             if (!dequeFrame()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
//         }
//     }

//     bool dequeFrame() override {
//         if (!streaming_) return false;
//         v4l2_buffer b{}; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP;
//         if (ioctl(fd_, VIDIOC_DQBUF, &b) < 0) return false;
//         auto& bf = bufs_[b.index];
//         const size_t bytes = b.bytesused;
//         auto bufferPtr = bufferManager_.acquireBuffer(b.index, bf.start);
//         uint64_t fid = frameCounter_.fetch_add(1);
//         auto zf = std::make_shared<ZeroCopyFrameData>(bufferPtr, bytes, cfg_.width, cfg_.height, b.index, fid);
//         if (dispQ_) dispQ_->push(zf);
//         if (algoQ_) algoQ_->push(zf);
//         fpsCount_++;
//         auto now = std::chrono::steady_clock::now();
//         double sec = std::chrono::duration<double>(now - fpsT0_).count();
//         if (sec >= 1.0) { lastFps_ = fpsCount_ / sec; fpsCount_ = 0; fpsT0_ = now; }
//         if (agg_) {
//             CameraStats cs; cs.timestamp = std::chrono::system_clock::now(); cs.frameNumber = fid;
//             cs.fps = lastFps_; cs.frameWidth = cfg_.width; cs.frameHeight = cfg_.height; cs.frameSize = bytes;
//             agg_->beginFrame(fid, cs);
//         }
//         return true;
//     }

//     double getLastFPS() override { return lastFps_; }
//     void unmapBuffers() {
//         for (auto& b : bufs_) if (b.start && b.start != MAP_FAILED) munmap(b.start, b.len);
//         bufs_.clear();
//     }
//     void reportError(const std::string& m) {
//         if (errCb_) errCb_(m); else spdlog::error("[DataConcrete] {}", m);
//     }
//     bool queueBuffer(size_t idx) override { return queueBufferInternal(idx); }
//     bool queueBufferInternal(size_t idx) {
//         v4l2_buffer b{}; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; b.index = idx;
//         if (ioctl(fd_, VIDIOC_QBUF, &b) < 0) return reportError("QBUF"), false;
//         bufs_[idx].st = BufState::QUEUED; return true;
//     }
//     void setErrorCallback(std::function<void(const std::string&)> cb) override { errCb_ = std::move(cb); }
//     int getQueueSize() const override { return 0; }

// private:
//     enum class BufState { FREE, QUEUED, IN_FLIGHT };
//     struct Buffer { void* start = nullptr; size_t len = 0; BufState st = BufState::FREE; };
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> algoQ_;
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> dispQ_;
//     std::shared_ptr<ISystemMetricsAggregator> agg_;
//     BufferManager bufferManager_;
//     int fd_ = -1;
//     CameraConfig cfg_;
//     std::vector<Buffer> bufs_;
//     static constexpr int NUM_BUF = 4;
//     std::atomic<bool> streaming_{false};
//     std::atomic<bool> running_{false};
//     std::thread capThread_;
//     std::atomic<uint64_t> frameCounter_{0};
//     std::atomic<int> fpsCount_{0};
//     std::chrono::steady_clock::time_point fpsT0_ = std::chrono::steady_clock::now();
//     double lastFps_ = 0.0;
//     std::mutex m_;
//     std::function<void(const std::string&)> errCb_;
// };
