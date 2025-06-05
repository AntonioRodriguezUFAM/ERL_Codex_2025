// SdlDisplayConcrete_new.h
// (Stage 2: Display Concrete Class)

// //=======================================================================
// #ifndef SDL_DISPLAY_CONCRETE_NEW_H
// #define SDL_DISPLAY_CONCRETE_NEW_H

// #include "../Interfaces/IDisplay.h"
// #include "../SharedStructures/DisplayConfig.h"
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/SharedQueue.h"

// #include <SDL2/SDL.h>
// #include <atomic>
// #include <vector>
// #include <mutex>
// #include <string>
// #include <memory>
// #include <functional>
// #include <stdexcept>
// #include <iostream>
// #include <spdlog/spdlog.h>

// /**
//  * @class SdlDisplayConcrete
//  * @brief Implements IDisplay using SDL2. Renders two streams side-by-side:
//  *        - Original frames (YUYV->RGB) from `originalQueue_`
//  *        - Processed frames (YUYV->RGB) from `processedQueue_`
//  */
// class SdlDisplayConcrete : public IDisplay {
// public:
//     SdlDisplayConcrete(std::shared_ptr<SharedQueue<FrameData>> origQueue,
//                        std::shared_ptr<SharedQueue<FrameData>> procQueue,
//                         ThreadManager& threadManager);
//     ~SdlDisplayConcrete() override;

//     bool configure(const DisplayConfig& config) override;
//     bool initializeDisplay(int width, int height) override;
//     void updateOriginalFrame(const uint8_t* rgbData, int width, int height) override;
//     void updateProcessedFrame(const uint8_t* rgbData, int width, int height) override;
//     void renderAndPollEvents() override;
//     void setErrorCallback(std::function<void(const std::string&)>) override;
//     void closeDisplay() override;
//     bool is_Running() override;

//     // YUYV -> RGB converter
//     static void yuyvToRGB(const uint8_t* yuyv, uint8_t* rgb, int width, int height);

// private:
//     void reportError(const std::string& msg);

//     // SDL resources
//     struct SDL_Deleter {
//         void operator()(SDL_Window* window) const   { if (window)   SDL_DestroyWindow(window); }
//         void operator()(SDL_Renderer* r) const      { if (r)        SDL_DestroyRenderer(r); }
//         void operator()(SDL_Texture* texture) const { if (texture)  SDL_DestroyTexture(texture); }
//     };

//     std::unique_ptr<SDL_Window, SDL_Deleter>   window_;
//     std::unique_ptr<SDL_Renderer, SDL_Deleter> renderer_;
//     std::unique_ptr<SDL_Texture, SDL_Deleter>  texOrig_;
//     std::unique_ptr<SDL_Texture, SDL_Deleter>  texProc_;

//     int  displayWidth_  = 0;
//     int  displayHeight_ = 0;
//     //bool initialized_{false};

//     std::atomic<bool> running_;
//     std::atomic<bool> initialized_{false};
//     std::function<void(const std::string&)> errorCallback_;

//     std::mutex originalMutex_;
//     std::mutex processedMutex_;

//     const DisplayConfig* config_ = nullptr;

//     // Shared queues
//     // Shared queues for original and processed frames
//     std::shared_ptr<SharedQueue<FrameData>> originalQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> processedQueue_;

    

//      // Add these member variables
//      FrameData lastOriginal_;
//      FrameData lastProcessed_;
// };

// // --------------------- Implementation --------------------- //

// ////////////////////////////////////////////////////////////////
// // Implementation
// ////////////////////////////////////////////////////////////////

// inline SdlDisplayConcrete::SdlDisplayConcrete(std::shared_ptr<SharedQueue<FrameData>> origQueue,
//                                               std::shared_ptr<SharedQueue<FrameData>> procQueue,
//                                                 ThreadManager& threadManager)
//     : running_(false)
//     , originalQueue_(origQueue)
//     , processedQueue_(procQueue)
// {
// }

// inline SdlDisplayConcrete::~SdlDisplayConcrete() {
//     closeDisplay();
// }

// inline void SdlDisplayConcrete::reportError(const std::string& msg) {
//     if (errorCallback_) {
//         errorCallback_(msg);
//     } else {
//         spdlog::error("[SdlDisplayConcrete] {}", msg);
//     }
// }

// inline bool SdlDisplayConcrete::configure(const DisplayConfig& config) {
//     if (config.width <= 0 || config.height <= 0) {
//         reportError("Invalid display configuration (width/height <= 0).");
//         return false;
//     }
//     config_ = &config;
//     return true;
// }

// inline bool SdlDisplayConcrete::initializeDisplay(int width, int height) {
//     if (initialized_) {
//         spdlog::warn("[SdlDisplayConcrete] Display is already initialized.");
//         return true;
//     }
//     if (SDL_Init(SDL_INIT_VIDEO) < 0) {
//         reportError("SDL_Init failed: " + std::string(SDL_GetError()));
//         return false;
//     }

//     // Use the DisplayConfig for window title and fullscreen flag.
//     Uint32 windowFlags = SDL_WINDOW_SHOWN;
//     if (config_ && config_->fullScreen) {
//         windowFlags |= SDL_WINDOW_FULLSCREEN;
//     }

//     // Use the DisplayConfig for window title and fullscreen flag.
//     window_.reset(SDL_CreateWindow(
//         (config_ && !config_->windowTitle.empty()) ? config_->windowTitle.c_str() : "ERL Display",
//         SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
//         width, height, SDL_WINDOW_SHOWN
//     ));
//     if (!window_) {
//         reportError("SDL_CreateWindow failed: " + std::string(SDL_GetError()));
//         SDL_Quit();
//         return false;
//     }

//     renderer_.reset(SDL_CreateRenderer(window_.get(), -1, SDL_RENDERER_ACCELERATED));
//     if (!renderer_) {
//         reportError("SDL_CreateRenderer failed: " + std::string(SDL_GetError()));
//         window_.reset();
//         SDL_Quit();
//         return false;
//     }

//     texOrig_.reset(SDL_CreateTexture(renderer_.get(), SDL_PIXELFORMAT_RGB24,
//                                      SDL_TEXTUREACCESS_STREAMING, width, height));
//     if (!texOrig_) {
//         reportError("SDL_CreateTexture (orig) failed: " + std::string(SDL_GetError()));
//         renderer_.reset();
//         window_.reset();
//         SDL_Quit();
//         return false;
//     }

//     texProc_.reset(SDL_CreateTexture(renderer_.get(), SDL_PIXELFORMAT_RGB24,
//                                      SDL_TEXTUREACCESS_STREAMING, width, height));
//     if (!texProc_) {
//         reportError("SDL_CreateTexture (proc) failed: " + std::string(SDL_GetError()));
//         texOrig_.reset();
//         renderer_.reset();
//         window_.reset();
//         SDL_Quit();
//         return false;
//     }

//     displayWidth_  = width;
//     displayHeight_ = height;
//     running_       = true;
//     initialized_   = true;

//     spdlog::info("[SdlDisplayConcrete] SDL display initialized: {}x{}", width, height);
//     return true;
// }

// inline void SdlDisplayConcrete::updateOriginalFrame(const uint8_t* rgbData, int width, int height) {
//     std::lock_guard<std::mutex> lock(originalMutex_);
//     if (texOrig_ && rgbData) {
//         SDL_UpdateTexture(texOrig_.get(), nullptr, rgbData, width * 3);
//     }
// }

// inline void SdlDisplayConcrete::updateProcessedFrame(const uint8_t* rgbData, int width, int height) {
//     std::lock_guard<std::mutex> lock(processedMutex_);
//     if (texProc_ && rgbData) {
//         SDL_UpdateTexture(texProc_.get(), nullptr, rgbData, width * 3);
//     }
// }


// inline void SdlDisplayConcrete::renderAndPollEvents() {
//     if (!running_) {
//         spdlog::debug("[SdlDisplayConcrete] renderAndPollEvents called but display is not running.");
//         return;
//     }
    
//     // Process SDL events
//     SDL_Event e;
//     while (SDL_PollEvent(&e)) {
//         if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)) {
//             spdlog::info("[SdlDisplayConcrete] Quit event received; stopping display.");
//             running_ = false;
//         }
//     }
//     if (!running_) {
//         return;
//     }

//     // Performance monitoring - start timer
//     static auto lastFrameTime = std::chrono::steady_clock::now();
//     auto frameStartTime = std::chrono::steady_clock::now();

//     // Maintain references to last valid frames
//     FrameData currentOriginal;
//     FrameData currentProcessed;

//     // Attempt to get new frames using non-blocking try_pop
//     FrameData newOriginal, newProcessed;
//     bool hasNewOriginal = originalQueue_->try_pop(newOriginal);
//     bool hasNewProcessed = processedQueue_->try_pop(newProcessed);

//     if (hasNewOriginal) {
//         currentOriginal = std::move(newOriginal);
//     } else {
//         // Use last known good frame or initialize if none exists
//         if (!lastOriginal_.dataVec.empty()) {
//             currentOriginal = lastOriginal_;
//         }
//     }

//     if (hasNewProcessed) {
//         currentProcessed = std::move(newProcessed);
//     } else {
//         // Use last known good frame or initialize if none exists
//         if (!lastProcessed_.dataVec.empty()) {
//             currentProcessed = lastProcessed_;
//         }
//     }

//     lastOriginal_ = std::move(currentOriginal);
//     lastProcessed_ = std::move(currentProcessed);

//     // Update textures with the latest frames
//     bool originalUpdated = false;
//     bool processedUpdated = false;

//     {
//         std::lock_guard<std::mutex> lock(originalMutex_);
//         if (!currentOriginal.dataVec.empty()) {
//             // Reuse buffer to avoid frequent allocations
//             static std::vector<uint8_t> rgbBuffer;
//             rgbBuffer.resize(currentOriginal.width * currentOriginal.height * 3);
//             yuyvToRGB(currentOriginal.dataVec.data(), rgbBuffer.data(), 
//                      currentOriginal.width, currentOriginal.height);
            
//             if (SDL_UpdateTexture(texOrig_.get(), nullptr, rgbBuffer.data(), 
//                                  currentOriginal.width * 3) != 0) {
//                 reportError("Failed to update original texture: " + std::string(SDL_GetError()));
//             } else {
//                 originalUpdated = true;
//             }
//         }
//     }

//     {
//         std::lock_guard<std::mutex> lock(processedMutex_);
//         if (!currentProcessed.dataVec.empty()) {
//             // Reuse buffer to avoid frequent allocations
//             static std::vector<uint8_t> rgbBuffer;
//             rgbBuffer.resize(currentProcessed.width * currentProcessed.height * 3);
//             yuyvToRGB(currentProcessed.dataVec.data(), rgbBuffer.data(), 
//                      currentProcessed.width, currentProcessed.height);
            
//             if (SDL_UpdateTexture(texProc_.get(), nullptr, rgbBuffer.data(), 
//                                  currentProcessed.width * 3) != 0) {
//                 reportError("Failed to update processed texture: " + std::string(SDL_GetError()));
//             } else {
//                 processedUpdated = true;
//             }
//         }
//     }

//     // Render
//     SDL_RenderClear(renderer_.get());

//     // Render original frame (left half) if updated
//     if (originalUpdated) {
//         SDL_Rect rectLeft = {0, 0, displayWidth_ / 2, displayHeight_};
//         if (SDL_RenderCopy(renderer_.get(), texOrig_.get(), nullptr, &rectLeft) != 0) {
//             reportError("Failed to render original frame: " + std::string(SDL_GetError()));
//         }
//     }

//     // Render processed frame (right half) if updated
//     if (processedUpdated) {
//         SDL_Rect rectRight = {displayWidth_ / 2, 0, displayWidth_ / 2, displayHeight_};
//         if (SDL_RenderCopy(renderer_.get(), texProc_.get(), nullptr, &rectRight) != 0) {
//             reportError("Failed to render processed frame: " + std::string(SDL_GetError()));
//         }
//     }

//     // if (SDL_RenderPresent(renderer_.get()) != 0) {
//     //     reportError("Failed to present render: " + std::string(SDL_GetError()));
//     // }

//       // Present the rendered frame
//     SDL_RenderPresent(renderer_.get());

//     // Performance monitoring - calculate and log render time
//     auto frameEndTime = std::chrono::steady_clock::now();
//     auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frameEndTime - frameStartTime).count();
//     spdlog::debug("Render time: {}ms", elapsed);

//     // Update last frame time for next iteration
//     lastFrameTime = frameEndTime;

//     // Control render loop frequency (~60 FPS)
//     std::this_thread::sleep_for(std::chrono::milliseconds(16));
// }


// inline void SdlDisplayConcrete::setErrorCallback(std::function<void(const std::string&)> callback) {
//     errorCallback_ = std::move(callback);
// }

// inline void SdlDisplayConcrete::closeDisplay() {
//     running_ = false;
//     texOrig_.reset();
//     texProc_.reset();
//     renderer_.reset();
//     window_.reset();
//     SDL_Quit();
//     initialized_ = false;
// }

// inline bool SdlDisplayConcrete::is_Running() {
//     return running_;
// }

// inline void SdlDisplayConcrete::yuyvToRGB(const uint8_t* yuyv, uint8_t* rgb, int width, int height) {
//     auto clamp = [](int v) { return (v < 0) ? 0 : ((v > 255) ? 255 : v); };
//     int size = width * height * 2;
//     for (int i = 0, j = 0; i < size; i += 4, j += 6) {
//         uint8_t Y0 = yuyv[i + 0];
//         uint8_t U  = yuyv[i + 1];
//         uint8_t Y1 = yuyv[i + 2];
//         uint8_t V  = yuyv[i + 3];

//         int C1 = static_cast<int>(Y0) - 16;
//         int D  = static_cast<int>(U) - 128;
//         int E  = static_cast<int>(V) - 128;

//         int R1 = clamp((298 * C1 + 409 * E + 128) >> 8);
//         int G1 = clamp((298 * C1 - 100 * D - 208 * E + 128) >> 8);
//         int B1 = clamp((298 * C1 + 516 * D + 128) >> 8);

//         rgb[j + 0] = static_cast<uint8_t>(R1);
//         rgb[j + 1] = static_cast<uint8_t>(G1);
//         rgb[j + 2] = static_cast<uint8_t>(B1);

//         int C2 = static_cast<int>(Y1) - 16;
//         int R2 = clamp((298 * C2 + 409 * E + 128) >> 8);
//         int G2 = clamp((298 * C2 - 100 * D - 208 * E + 128) >> 8);
//         int B2 = clamp((298 * C2 + 516 * D + 128) >> 8);

//         rgb[j + 3] = static_cast<uint8_t>(R2);
//         rgb[j + 4] = static_cast<uint8_t>(G2);
//         rgb[j + 5] = static_cast<uint8_t>(B2);
//     }
// }

// #endif // SDL_DISPLAY_CONCRETE_NEW_H

//=============================working code !! Final version==========================================

// // SdlDisplayConcrete_new.h
// #ifndef SDL_DISPLAY_CONCRETE_NEW_H
// #define SDL_DISPLAY_CONCRETE_NEW_H

// #include "../Interfaces/IDisplay.h"
// #include "../SharedStructures/DisplayConfig.h"
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/SharedQueue.h"

// #include "../SharedStructures/ThreadManager.h"  // If needed for concurrency tasks

// #include <SDL2/SDL.h>
// #include <atomic>
// #include <vector>
// #include <mutex>
// #include <memory>
// #include <chrono>
// #include <spdlog/spdlog.h>

// class SdlDisplayConcrete : public IDisplay {
// public:
//     SdlDisplayConcrete(
//         std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> origQueue,
//         std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> procQueue
//     );
//     ~SdlDisplayConcrete() override;

//     bool configure(const DisplayConfig& config) override;
//     bool initializeDisplay(int width, int height) override;
//      // Implementations for pure virtual functions from IDisplay

//     //virtual void updateOriginalFrame(const uint8_t* rgbData, int width, int height) override; // Add this
    
//     virtual void updateOriginalFrame(const uint8_t* rgbData, int width, [[maybe_unused]] int height)override; // Add this
//     virtual void updateProcessedFrame(const uint8_t* rgbData, int width, [[maybe_unused]] int height) override; // Add this


//     //virtual void updateProcessedFrame(const uint8_t* rgbData, int width, int height) override; // Add this
    
//     void renderAndPollEvents() override;
//     void setErrorCallback(std::function<void(const std::string&)>) override;
//     void closeDisplay() override;
//     //bool is_Running() const override;
//     bool is_Running() override;

//      // Method to update texture with new frame data
//      void updateTexture(SDL_Texture* tex, const std::shared_ptr<ZeroCopyFrameData>& frame, std::vector<uint8_t>& buffer);

//     static void yuyvToRGB(const uint8_t* yuyv, uint8_t* rgb, int width, int height);

// private:
//     struct SDL_Deleter {
//         void operator()(SDL_Window* window) const { if (window) SDL_DestroyWindow(window); }
//         void operator()(SDL_Renderer* r) const { if (r) SDL_DestroyRenderer(r); }
//         void operator()(SDL_Texture* texture) const { if (texture) SDL_DestroyTexture(texture); }
//     };

//     void reportError(const std::string& msg);
//     void initializeTextures(int width, int height);
//     void processEvents();
//     //void updateTexture(SDL_Texture* tex, const FrameData& frame, std::vector<uint8_t>& buffer);
//     //void updateTexture(SDL_Texture* tex, const std::shared_ptr<ZeroCopyFrameData>& frame, std::vector<uint8_t>& buffer);

//     void renderFrame();
//     void adaptiveSleep();
//     void handleSDLError(const std::string& operation);

//     //static void yuyvToRGB(const uint8_t* yuyv, uint8_t* rgb, int width, int height);

//     std::unique_ptr<SDL_Window, SDL_Deleter>   window_;
//     std::unique_ptr<SDL_Renderer, SDL_Deleter> renderer_;
//     std::unique_ptr<SDL_Texture, SDL_Deleter>  texOrig_;
//     std::unique_ptr<SDL_Texture, SDL_Deleter>  texProc_;

//     std::atomic<bool> running_{false};
//     std::atomic<bool> initialized_{false};
//     std::mutex stateMutex_;

//     DisplayConfig config_;
//     int displayWidth_{0};
//     int displayHeight_{0};

//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> originalQueue_;
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> processedQueue_;

//     std::shared_ptr<ZeroCopyFrameData> lastOriginal_;
//     std::shared_ptr<ZeroCopyFrameData> lastProcessed_;

//     std::vector<uint8_t> origRgbBuffer_;
//     std::vector<uint8_t> procRgbBuffer_;


//     std::chrono::steady_clock::time_point lastFrameTime_;
//     double averageFrameTime_{0.0};
//     int frameCounter_{0};
//     std::chrono::steady_clock::time_point performanceCheckTime_;

//     std::function<void(const std::string&)> errorCallback_;
// };


// // --------------------- Implementation --------------------- //

// // ////////////////////////////////////////////////////////////////
// // // Implementation
// // ////////////////////////////////////////////////////////////////

// SdlDisplayConcrete::SdlDisplayConcrete(
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> origQueue,
//     std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> procQueue
// ) : originalQueue_(std::move(origQueue)),
//     processedQueue_(std::move(procQueue)),
//     lastFrameTime_(std::chrono::steady_clock::now()),
//     performanceCheckTime_(std::chrono::steady_clock::now())
// {
//     // Initialize with empty frames
//     // lastOriginal_ = std::make_shared<ZeroCopyFrameData>();
//     // lastOriginal_->width = 640;
//     // lastOriginal_->height = 480;
//     // lastProcessed_ = lastOriginal_;

// // Initialize with dummy frame (provide all parameters)
//     auto dummyBuffer = std::shared_ptr<void>(); // Empty bufferData
//     lastOriginal_ = std::make_shared<ZeroCopyFrameData>(
//         dummyBuffer,
//         0,          // size
//         640,        // width
//         480,        // height
//         0,          // bufferIndex
//         0           // frameNumber
//     );
//     lastProcessed_ = lastOriginal_;

// }

// SdlDisplayConcrete::~SdlDisplayConcrete() {
//     closeDisplay();
// }

// void SdlDisplayConcrete::reportError(const std::string& msg) {
//     if (errorCallback_) {
//         errorCallback_(msg);
//     } else {
//         spdlog::error("[SdlDisplayConcrete] {}", msg);
//     }
// }

// bool SdlDisplayConcrete::configure(const DisplayConfig& config) {
//     std::lock_guard<std::mutex> lock(stateMutex_);
    
//     if (config.width <= 0 || config.height <= 0) {
//         reportError("Invalid display configuration (width/height <= 0).");
//         return false;
//     }
//     config_ = config;
//     return true;
// }

// void SdlDisplayConcrete::updateOriginalFrame(const uint8_t* rgbData, int width, [[maybe_unused]] int height) {
//     std::lock_guard<std::mutex> lock(stateMutex_);
//     if (texOrig_ && rgbData) {
//         if (SDL_UpdateTexture(texOrig_.get(), nullptr, rgbData, width * 3) != 0) {
//             reportError("Failed to update original texture: " + std::string(SDL_GetError()));
//         }
//     }
// }


// void SdlDisplayConcrete::updateProcessedFrame(const uint8_t* rgbData, int width, [[maybe_unused]] int height) {
//     std::lock_guard<std::mutex> lock(stateMutex_);
//     if (texProc_ && rgbData) {
//         if (SDL_UpdateTexture(texProc_.get(), nullptr, rgbData, width * 3) != 0) {
//             reportError("Failed to update processed texture: " + std::string(SDL_GetError()));
//         }
//     }
// }


// bool SdlDisplayConcrete::initializeDisplay(int width, int height) {
//     if (initialized_) {
//         spdlog::warn("[SdlDisplayConcrete] Display is already initialized.");
//         return true;
//     }

//     if (SDL_Init(SDL_INIT_VIDEO) < 0) {
//         reportError("SDL_Init failed: " + std::string(SDL_GetError()));
//         return false;
//     }

//     Uint32 windowFlags = SDL_WINDOW_SHOWN;
//     if (config_.fullScreen) {
//         windowFlags |= SDL_WINDOW_FULLSCREEN;
//     }

//     window_.reset(SDL_CreateWindow(
//         config_.windowTitle.empty() ? "ERL Display" : config_.windowTitle.c_str(),
//         SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
//         width, height, windowFlags
//     ));
//     if (!window_) {
//         reportError("SDL_CreateWindow failed: " + std::string(SDL_GetError()));
//         SDL_Quit();
//         return false;
//     }

//     renderer_.reset(SDL_CreateRenderer(window_.get(), -1, 
//         SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC));
//     if (!renderer_) {
//         reportError("SDL_CreateRenderer failed: " + std::string(SDL_GetError()));
//         window_.reset();
//         SDL_Quit();
//         return false;
//     }

//     initializeTextures(width, height);
//     initialized_ = true;
//     running_ = true;
//     displayWidth_ = width;
//     displayHeight_ = height;
    
//     spdlog::info("[SdlDisplayConcrete] SDL display initialized: {}x{}", width, height);

//     // Create textures with valid dimensions
//     if (!texOrig_ || !texProc_) {
//         reportError("Texture creation failed");
//         return false;
//     }

//     // Set texture dimensions explicitly
//     int textureWidth, textureHeight;
//     SDL_QueryTexture(texOrig_.get(), nullptr, nullptr, &textureWidth, &textureHeight);
//     if (textureWidth != width || textureHeight != height) {
//         reportError("Texture dimensions mismatch");
//         return false;
//     }


//     return true;
// }

// void SdlDisplayConcrete::initializeTextures(int width, int height) {
//     texOrig_.reset(SDL_CreateTexture(
//         renderer_.get(),
//         SDL_PIXELFORMAT_RGB24,
//         SDL_TEXTUREACCESS_STREAMING,
//         width, height
//     ));
    
//     texProc_.reset(SDL_CreateTexture(
//         renderer_.get(),
//         SDL_PIXELFORMAT_RGB24,
//         SDL_TEXTUREACCESS_STREAMING,
//         width, height
//     ));

//     if (!texOrig_ || !texProc_) {
//         reportError("Texture creation failed");
//         throw std::runtime_error("Failed to create textures");
//     }
// }




// //=======================================================================================
// // SdlDisplayConcrete.cpp
// //======================================================================================
// void SdlDisplayConcrete::renderAndPollEvents() {
//     spdlog::debug("[SdlDisplayConcrete] Entering renderAndPollEvents");

//     // 1. Check if display is still running
//     if (!running_) {
//         spdlog::debug("[SdlDisplayConcrete] renderAndPollEvents called but display is not running. Exiting early.");
//         return;
//     }

//     // 2. Handle SDL events
//     SDL_Event e;
//     while (SDL_PollEvent(&e)) {
//         if (e.type == SDL_QUIT || 
//             (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)) {
//             spdlog::info("[SdlDisplayConcrete] Quit event received; stopping display.");
//             running_ = false;
//         }
//     }
//     if (!running_) {
//         spdlog::debug("[SdlDisplayConcrete] Display flagged as not running after event processing. Exiting.");
//         return;
//     }

//     auto frameStartTime = std::chrono::steady_clock::now();

//     // 3. Attempt to pop new frames from each queue
//     spdlog::debug("[SdlDisplayConcrete] Attempting to pop frames from original & processed queues.");
//     std::shared_ptr<ZeroCopyFrameData> newOriginal, newProcessed;
//     bool hasNewOriginal = false;
//     bool hasNewProcessed = false;

//     // Validate queues before accessing
//     if (originalQueue_) {
//         hasNewOriginal = originalQueue_->try_pop(newOriginal);
//     } else {
//         spdlog::warn("[SdlDisplayConcrete] originalQueue_ is null! Skipping original frame processing.");
//     }

//     if (processedQueue_) {
//         hasNewProcessed = processedQueue_->try_pop(newProcessed);
//     } else {
//         spdlog::debug("[SdlDisplayConcrete] processedQueue_ is null, skipping processed frame.");
//     }

//     // Determine current frames with validity checks
//     std::shared_ptr<ZeroCopyFrameData> currentOriginal = hasNewOriginal ? newOriginal : lastOriginal_;
//     std::shared_ptr<ZeroCopyFrameData> currentProcessed = hasNewProcessed ? newProcessed : lastProcessed_;

//     // Check if frames are valid
//     if (currentOriginal && currentOriginal->bufferData && currentOriginal->dataPtr) {
//         // Process original frame
//     } else {
//         spdlog::warn("Skipping invalid original frame");
//     }

//     if (currentProcessed && currentProcessed->bufferData && currentProcessed->dataPtr) {
//         // Process processed frame
//     } else {
//         spdlog::warn("Skipping invalid processed frame");
//     }

//     // Update last frames only if valid
//     bool validOriginal = currentOriginal && currentOriginal->dataPtr;
//     bool validProcessed = currentProcessed && currentProcessed->dataPtr;

//     if (validOriginal) {
//         lastOriginal_ = currentOriginal;
//     } else {
//         currentOriginal = lastOriginal_;
//         validOriginal = currentOriginal && currentOriginal->dataPtr;
//     }

//     if (validProcessed) {
//         lastProcessed_ = currentProcessed;
//     } else {
//         currentProcessed = lastProcessed_;
//         validProcessed = currentProcessed && currentProcessed->dataPtr;
//     }

//     // Log frame validity
//     spdlog::debug("[SdlDisplayConcrete] hasNewOriginal={}, validOriginal={}, hasNewProcessed={}, validProcessed={}",
//         hasNewOriginal, validOriginal, hasNewProcessed, validProcessed);

//     // 4. Convert & update textures if frames are valid
//     bool originalUpdated = false;
//     bool processedUpdated = false;

//     // 4a. Original frame processing
//     {
//         std::lock_guard<std::mutex> lock(stateMutex_);
//         if (validOriginal) {
//             spdlog::debug("[SdlDisplayConcrete] Processing original frame: {}x{}", 
//                          currentOriginal->width, currentOriginal->height);

//             // Convert YUYV to RGB
//             origRgbBuffer_.resize(currentOriginal->width * currentOriginal->height * 3);
//             yuyvToRGB(static_cast<uint8_t*>(currentOriginal->dataPtr),
//                       origRgbBuffer_.data(),
//                       currentOriginal->width,
//                       currentOriginal->height);

//             // Update texture
//             if (texOrig_) {
//                 if (SDL_UpdateTexture(texOrig_.get(), nullptr, origRgbBuffer_.data(),
//                                      currentOriginal->width * 3) == 0) {
//                     originalUpdated = true;
//                 } else {
//                     reportError("Failed to update original texture: " + std::string(SDL_GetError()));
//                 }
//             }
//         } else {
//             spdlog::warn("[SdlDisplayConcrete] Skipping invalid original frame");
//         }
//     }

//     // 4b. Processed frame processing
//     {
//         std::lock_guard<std::mutex> lock(stateMutex_);
//         if (validProcessed) {
//             spdlog::debug("[SdlDisplayConcrete] Processing processed frame: {}x{}", 
//                          currentProcessed->width, currentProcessed->height);

//             // Convert YUYV to RGB
//             procRgbBuffer_.resize(currentProcessed->width * currentProcessed->height * 3);
//             yuyvToRGB(static_cast<uint8_t*>(currentProcessed->dataPtr),
//                       procRgbBuffer_.data(),
//                       currentProcessed->width,
//                       currentProcessed->height);

//             // Update texture
//             if (texProc_) {
//                 if (SDL_UpdateTexture(texProc_.get(), nullptr, procRgbBuffer_.data(),
//                                      currentProcessed->width * 3) == 0) {
//                     processedUpdated = true;
//                 } else {
//                     reportError("Failed to update processed texture: " + std::string(SDL_GetError()));
//                 }
//             }
//         } else {
//             spdlog::warn("[SdlDisplayConcrete] Skipping invalid processed frame");
//         }
//     }

//     // 5. Render
//     if (!renderer_) {
//         spdlog::error("[SdlDisplayConcrete] renderer_ is null! Rendering will fail.");
//         return;
//     }

//     // Clear and render
//     SDL_RenderClear(renderer_.get());

//     if (originalUpdated) {
//         SDL_Rect rectLeft = {0, 0, displayWidth_ / 2, displayHeight_};
//         SDL_RenderCopy(renderer_.get(), texOrig_.get(), nullptr, &rectLeft);
//     }

//     if (processedUpdated) {
//         SDL_Rect rectRight = {static_cast<int>(displayWidth_ / 2), 0, displayWidth_ / 2, displayHeight_};
//         SDL_RenderCopy(renderer_.get(), texProc_.get(), nullptr, &rectRight);
//     }

//     SDL_RenderPresent(renderer_.get());

//     // 6. Performance logging
//     auto frameEndTime = std::chrono::steady_clock::now();
//     auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frameEndTime - frameStartTime).count();
//     spdlog::debug("[SdlDisplayConcrete] Frame took {}ms", elapsed);

//     // Update FPS calculation
//     frameCounter_++;
//     if (frameCounter_ % 60 == 0) {
//         auto now = std::chrono::steady_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - performanceCheckTime_).count();
//         double fps = 60.0 * 1000.0 / duration;
//         spdlog::info("[SdlDisplayConcrete] Average FPS: {:.1f}", fps);
//         performanceCheckTime_ = now;
//     }

//     // 7. Throttle to ~60 FPS
//     std::this_thread::sleep_for(std::chrono::milliseconds(16));
//     spdlog::debug("[SdlDisplayConcrete] Exiting renderAndPollEvents");
// }



// // //=======================================================================================
// // void SdlDisplayConcrete::renderAndPollEvents() {
// //     spdlog::debug("[SdlDisplayConcrete] Entering renderAndPollEvents");

// //     // 1. Check if display is still running
// //     if (!running_) {
// //         spdlog::debug("[SdlDisplayConcrete] renderAndPollEvents called but display is not running. Exiting early.");
// //         return;
// //     }

// //     // 2. Handle SDL events
// //     SDL_Event e;
// //     while (SDL_PollEvent(&e)) {
// //         if (e.type == SDL_QUIT || 
// //            (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)) {
// //             spdlog::info("[SdlDisplayConcrete] Quit event received; stopping display.");
// //             running_ = false;
// //         }
// //     }
// //     if (!running_) {
// //         spdlog::debug("[SdlDisplayConcrete] Display flagged as not running after event processing. Exiting.");
// //         return;
// //     }

// //     auto frameStartTime = std::chrono::steady_clock::now();

// //     // 3. Attempt to pop new frames from each queue
// //     spdlog::debug("[SdlDisplayConcrete] Attempting to pop frames from original & processed queues.");
// //     std::shared_ptr<ZeroCopyFrameData> newOriginal, newProcessed;

// //     bool hasNewOriginal = false;
// //     bool hasNewProcessed = false;

    
// //     //bool hasNewOriginal   = originalQueue_->try_pop(newOriginal);
// //     //bool hasNewProcessed  = processedQueue_->try_pop(newProcessed);

// //     // Check if originalQueue_ is valid before popping
// //     // Check Queue Validity Before Access
// //     if (originalQueue_) {
// //         hasNewOriginal = originalQueue_->try_pop(newOriginal);
// //     } else {
// //         spdlog::warn("[SdlDisplayConcrete] originalQueue_ is null!");
// //         return;
// //     }

// //     // Check if processedQueue_ is valid before popping
// //     if (processedQueue_) {
// //         hasNewProcessed = processedQueue_->try_pop(newProcessed);
// //     } else {
// //         spdlog::debug("[SdlDisplayConcrete] processedQueue_ is null, skipping.");
// //         return;
// //     }
// //     //=========

// //     if (!originalQueue_ || !processedQueue_) {
// //         spdlog::error("[SdlDisplayConcrete] originalQueue_ or processedQueue_ is null! This will cause undefined behavior.");
// //     }
    
// // // //======================================================================================================
// // //     if (newProcessed && !newProcessed->dataPtr) {
// // //         spdlog::warn("[SdlDisplayConcrete] Processed frame has invalid dataPtr. Skipping update.");
// // //         return;
// // //     }
   
// // // //======================================================================================================

// //     spdlog::debug("[SdlDisplayConcrete] hasNewOriginal={}, hasNewProcessed={}", 
// //                   hasNewOriginal, hasNewProcessed);

// //     // 3a. Decide which frames to use this iteration
// //     std::shared_ptr<ZeroCopyFrameData> currentOriginal  = hasNewOriginal ? newOriginal : lastOriginal_;
// //     std::shared_ptr<ZeroCopyFrameData> currentProcessed = hasNewProcessed ? newProcessed : lastProcessed_;

// //     lastOriginal_  = currentOriginal;   // Keep references to last frames
// //     lastProcessed_ = currentProcessed;

// //     // 4. Convert & update textures if frames are valid
// //     bool originalUpdated  = false;
// //     bool processedUpdated = false;

// //     // 4a. Original frame block
// //     {
// //         std::lock_guard<std::mutex> lock(stateMutex_);
// //         spdlog::debug("[SdlDisplayConcrete] Checking currentOriginal for YUYV->RGB update.");
// //         if (currentOriginal) {
// //             if (currentOriginal->dataPtr) {
// //                 spdlog::debug("[SdlDisplayConcrete] currentOriginal frame#: {}, w={}, h={}, size={}", 
// //                               currentOriginal->frameNumber,
// //                               currentOriginal->width, 
// //                               currentOriginal->height, 
// //                               currentOriginal->size);
// //                 // Resize buffer to hold RGB data
// //                 origRgbBuffer_.resize(currentOriginal->width * currentOriginal->height * 3);

// //                 // Convert from YUYV to RGB
// //                 yuyvToRGB(static_cast<uint8_t*>(currentOriginal->dataPtr),
// //                           origRgbBuffer_.data(),
// //                           currentOriginal->width,
// //                           currentOriginal->height);

// //                 // Update the SDL texture
// //                 if (!texOrig_) {
// //                     spdlog::error("[SdlDisplayConcrete] texOrig_ is null! Did you initialize textures properly?");
// //                 } else {
// //                     if (SDL_UpdateTexture(texOrig_.get(), nullptr, origRgbBuffer_.data(),
// //                                          currentOriginal->width * 3) != 0) {
// //                         reportError("Failed to update original texture: " + std::string(SDL_GetError()));
// //                     } else {
// //                         originalUpdated = true;
// //                         spdlog::debug("[SdlDisplayConcrete] Original texture updated successfully.");
// //                     }
// //                 }
// //             } else {
// //                 spdlog::warn("[SdlDisplayConcrete] currentOriginal is non-null but dataPtr is null!");
// //             }
// //         } else {
// //             spdlog::debug("[SdlDisplayConcrete] currentOriginal is null. Using last frame or none at all.");
// //         }
// //     }

// //     // 4b. Processed frame block
// //     {
// //         std::lock_guard<std::mutex> lock(stateMutex_);
// //         spdlog::debug("[SdlDisplayConcrete] Checking currentProcessed for YUYV->RGB update.");
// //         if (currentProcessed) {
// //             if (currentProcessed->dataPtr) {
// //                 spdlog::debug("[SdlDisplayConcrete] currentProcessed frame#: {}, w={}, h={}, size={}", 
// //                               currentProcessed->frameNumber, 
// //                               currentProcessed->width,
// //                               currentProcessed->height, 
// //                               currentProcessed->size);
// //                 // Resize buffer for RGB
// //                 procRgbBuffer_.resize(currentProcessed->width * currentProcessed->height * 3);

// //                 // Convert
// //                 yuyvToRGB(static_cast<uint8_t*>(currentProcessed->dataPtr),
// //                           procRgbBuffer_.data(),
// //                           currentProcessed->width,
// //                           currentProcessed->height);

// //                 // Update processed texture
// //                 if (!texProc_) {
// //                     spdlog::error("[SdlDisplayConcrete] texProc_ is null! Did you initialize textures properly?");
// //                 } else {
// //                     if (SDL_UpdateTexture(texProc_.get(), nullptr, procRgbBuffer_.data(),
// //                                          currentProcessed->width * 3) != 0) {
// //                         reportError("Failed to update processed texture: " + std::string(SDL_GetError()));
// //                     } else {
// //                         processedUpdated = true;
// //                         spdlog::debug("[SdlDisplayConcrete] Processed texture updated successfully.");
// //                     }
// //                 }
// //             } else {
// //                 spdlog::warn("[SdlDisplayConcrete] currentProcessed is non-null but dataPtr is null!");
// //             }
// //         } else {
// //             spdlog::debug("[SdlDisplayConcrete] currentProcessed is null. Using last processed frame or none at all.");
// //         }
// //     }

// //     // 5. Render
// //     if (!renderer_) {
// //         spdlog::error("[SdlDisplayConcrete] renderer_ is null! Rendering will fail.");
// //     }

// //     spdlog::debug("[SdlDisplayConcrete] Clearing renderer...");
// //     SDL_RenderClear(renderer_.get());

// //     if (originalUpdated) {
// //         SDL_Rect rectLeft = {0, 0, displayWidth_ / 2, displayHeight_};
// //         if (SDL_RenderCopy(renderer_.get(), texOrig_.get(), nullptr, &rectLeft) != 0) {
// //             reportError("Failed to render original frame: " + std::string(SDL_GetError()));
// //         } else {
// //             spdlog::debug("[SdlDisplayConcrete] Rendered original frame to left side.");
// //         }
// //     }

// //     if (processedUpdated) {
// //         SDL_Rect rectRight = {static_cast<int>(displayWidth_ / 2), 0, displayWidth_ / 2, displayHeight_};
// //         if (SDL_RenderCopy(renderer_.get(), texProc_.get(), nullptr, &rectRight) != 0) {
// //             reportError("Failed to render processed frame: " + std::string(SDL_GetError()));
// //         } else {
// //             spdlog::debug("[SdlDisplayConcrete] Rendered processed frame to right side.");
// //         }
// //     }

// //     SDL_RenderPresent(renderer_.get());
// //     spdlog::debug("[SdlDisplayConcrete] SDL_RenderPresent called.");

// //     // 6. Performance logging
// //     auto frameEndTime = std::chrono::steady_clock::now();
// //     auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frameEndTime - frameStartTime).count();
// //     spdlog::debug("[SdlDisplayConcrete] Render time: {}ms", elapsed);

// //     frameCounter_++;
// //     if (frameCounter_ % 60 == 0) {
// //         auto now = std::chrono::steady_clock::now();
// //         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - performanceCheckTime_).count();
// //         double fps = 60.0 * 1000.0 / duration;
// //         spdlog::info("[SdlDisplayConcrete] Average FPS: {:.1f}", fps);
// //         performanceCheckTime_ = now;
// //     }

// //     // 7. Small sleep => ~60 FPS
// //     std::this_thread::sleep_for(std::chrono::milliseconds(16));
// //     spdlog::debug("[SdlDisplayConcrete] Exiting renderAndPollEvents");
// // }


// void SdlDisplayConcrete::setErrorCallback(std::function<void(const std::string&)> callback) {
//     errorCallback_ = std::move(callback);
// }

// void SdlDisplayConcrete::closeDisplay() {
//     running_ = false;
//     texOrig_.reset();
//     texProc_.reset();
//     renderer_.reset();
//     window_.reset();
//     SDL_Quit();
//     initialized_ = false;
//     spdlog::info("[SdlDisplayConcrete] Display closed.");
// }

// bool SdlDisplayConcrete::is_Running() {
//     return running_;
// }


// //================================================================0
// void SdlDisplayConcrete::updateTexture(SDL_Texture* tex, 
//     const std::shared_ptr<ZeroCopyFrameData>& frame, 
//     std::vector<uint8_t>& buffer) {
// // Validate inputs
// if (!tex || !frame || !frame->dataPtr) {
// reportError("Invalid texture or frame data in updateTexture");
// return;
// }

// // Check frame dimensions for validity
// if (frame->width <= 0 || frame->height <= 0) {
// reportError("Frame has invalid dimensions: " + 
// std::to_string(frame->width) + "x" + std::to_string(frame->height));
// return;
// }

// // Calculate required buffer size
// size_t requiredBufferSize = frame->width * frame->height * 3; // RGB24 format
// if (buffer.size() != requiredBufferSize) {
// spdlog::warn("Resizing buffer from {} to {} bytes", buffer.size(), requiredBufferSize);
// buffer.resize(requiredBufferSize);
// }

// // Convert YUYV to RGB24
// try {
// yuyvToRGB(static_cast<uint8_t*>(frame->dataPtr),
// buffer.data(),
// frame->width,
// frame->height);
// } catch (const std::exception& e) {
// reportError("YUYV to RGB conversion failed: " + std::string(e.what()));
// return;
// }

// // Check texture dimensions match frame
// int textureW, textureH;
// if (SDL_QueryTexture(tex, nullptr, nullptr, &textureW, &textureH) != 0) {
// reportError("Failed to query texture dimensions: " + std::string(SDL_GetError()));
// return;
// }
// if (textureW != frame->width || textureH != frame->height) {
// reportError("Texture dimensions (" + std::to_string(textureW) + "x" + std::to_string(textureH) + 
// ") don't match frame (" + std::to_string(frame->width) + "x" + std::to_string(frame->height) + ")");
// return;
// }

// // Update texture
// if (SDL_UpdateTexture(tex, nullptr, buffer.data(), frame->width * 3) != 0) {
// reportError("Failed to update texture: " + std::string(SDL_GetError()));
// return;
// }

// spdlog::trace("Texture updated successfully for {}x{} frame", frame->width, frame->height);
// }
// //===================================================================

// // Method to update texture with new frame data
// // void SdlDisplayConcrete::updateTexture(SDL_Texture* tex, const std::shared_ptr<ZeroCopyFrameData>& frame, std::vector<uint8_t>& buffer) {
// //     // Check if the texture and frame data are valid
// //     if (!tex || !frame || !frame->dataPtr) {
// //         reportError("Invalid texture or frame data in updateTexture");
// //         return;
// //     }

// //     // Convert YUYV format to RGB24 format
// //     yuyvToRGB(static_cast<uint8_t*>(frame->dataPtr), buffer.data(), frame->width, frame->height);

// //     // Update the texture with the converted RGB data
// //     if (SDL_UpdateTexture(tex, nullptr, buffer.data(), frame->width * 3) != 0) {
// //         reportError("Failed to update texture: " + std::string(SDL_GetError()));
// //     }
// // }

// void SdlDisplayConcrete::yuyvToRGB(const uint8_t* yuyv, uint8_t* rgb, int width, int height) {
//     auto clamp = [](int v) { return (v < 0) ? 0 : ((v > 255) ? 255 : v); };
//     int size = width * height * 2;
//     for (int i = 0, j = 0; i < size; i += 4, j += 6) {
//         uint8_t Y0 = yuyv[i + 0];
//         uint8_t U  = yuyv[i + 1];
//         uint8_t Y1 = yuyv[i + 2];
//         uint8_t V  = yuyv[i + 3];

//         int C1 = static_cast<int>(Y0) - 16;
//         int D  = static_cast<int>(U) - 128;
//         int E  = static_cast<int>(V) - 128;

//         int R1 = clamp((298 * C1 + 409 * E + 128) >> 8);
//         int G1 = clamp((298 * C1 - 100 * D - 208 * E + 128) >> 8);
//         int B1 = clamp((298 * C1 + 516 * D + 128) >> 8);

//         rgb[j + 0] = static_cast<uint8_t>(R1);
//         rgb[j + 1] = static_cast<uint8_t>(G1);
//         rgb[j + 2] = static_cast<uint8_t>(B1);

//         int C2 = static_cast<int>(Y1) - 16;
//         int R2 = clamp((298 * C2 + 409 * E + 128) >> 8);
//         int G2 = clamp((298 * C2 - 100 * D - 208 * E + 128) >> 8);
//         int B2 = clamp((298 * C2 + 516 * D + 128) >> 8);

//         rgb[j + 3] = static_cast<uint8_t>(R2);
//         rgb[j + 4] = static_cast<uint8_t>(G2);
//         rgb[j + 5] = static_cast<uint8_t>(B2);
//     }
// }

// #endif // SDL_DISPLAY_CONCRETE_NEW_H



//=========================================================================

///=====================TESTING  code Final Version ==============================


// SdlDisplayConcrete_new.h
#ifndef SDL_DISPLAY_CONCRETE_NEW_H
#define SDL_DISPLAY_CONCRETE_NEW_H

//#pragma once

#include "../Interfaces/IDisplay.h"
#include "../SharedStructures/DisplayConfig.h"
#include "../SharedStructures/ZeroCopyFrameData.h"
#include "../SharedStructures/SharedQueue.h"
#include "../SharedStructures/ThreadManager.h"


#include "../Interfaces/ISystemMetricsAggregator.h"
#include "../SharedStructures/allModulesStatcs.h"               // The structure shown above


#include <SDL2/SDL.h>
#include <atomic>
#include <mutex>
#include <memory>
#include <chrono>
#include <functional>
#include <vector>
#include <spdlog/spdlog.h>

/**
 * @class SdlDisplayConcrete
 * @brief Displays frames using SDL. Optionally shows two frames side by side: original and processed.
 *
 * Internally uses SDL_Window, SDL_Renderer, and SDL_Texture for each frame (original & processed).
 * The frames are read from two SharedQueues of std::shared_ptr<ZeroCopyFrameData>:
 *   - originalQueue_  (for the unprocessed camera frames)
 *   - processedQueue_ (for the processed / algorithm output frames)
 *
 * The user calls renderAndPollEvents() in a loop to:
 *   1. Poll SDL events (close window, etc.)
 *   2. Pop from each queue (if available), convert from YUYV->RGB
 *   3. Update the SDL textures and render
 *
 * The display remains active until is_Running() is false or closeDisplay() is called.
 */
class SdlDisplayConcrete : public IDisplay {
public:
    /**
     * @brief Constructor taking queues for original and processed frames
     * @param origQueue SharedQueue of ZeroCopyFrameData for original frames
     * @param procQueue SharedQueue of ZeroCopyFrameData for processed frames
     */
    SdlDisplayConcrete(
        std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> origQueue,
        std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> procQueue,
        std::shared_ptr<ISystemMetricsAggregator> aggregator  // DataConcrete to support SystemMetricsAggregatorImpl injection and push camera metrics in real-time using the aggregator
    );

    /**
     * @brief Destructor (calls closeDisplay if still running)
     */
    ~SdlDisplayConcrete() override;

    // ---------------- IDisplay Interface ----------------

    /**
     * @brief Configure the display parameters (width, height, etc.)
     * @param config DisplayConfig structure
     * @return True on success, false otherwise
     */
    bool configure(const DisplayConfig& config) override;

    /**
     * @brief Initialize SDL and create a window/renderer
     * @param width  Desired window width
     * @param height Desired window height
     * @return True on success, false otherwise
     */
    bool initializeDisplay(int width, int height) override;

    /**
     * @brief Update the original-frame texture with raw RGB data
     * @param rgbData Pointer to RGB24 data
     * @param width   Frame width
     * @param height  Frame height
     */
    void updateOriginalFrame(const uint8_t* rgbData, int width, [[maybe_unused]]int height) override;

    /**
     * @brief Update the processed-frame texture with raw RGB data
     * @param rgbData Pointer to RGB24 data
     * @param width   Frame width
     * @param height  Frame height
     */
    void updateProcessedFrame(const uint8_t* rgbData, int width, [[maybe_unused]]int height) override;

    /**
     * @brief Render the current textures to the window and process SDL events
     */
    void renderAndPollEvents() override;

    /**
     * @brief Set an optional error callback
     * @param callback A function taking a const std::string&
     */
    void setErrorCallback(std::function<void(const std::string&)>) override;

    /**
     * @brief Closes the SDL display (destroys window, textures, etc.)
     */
    void closeDisplay() override;

    /**
     * @brief Returns whether the SDL display is still running
     */
    bool is_Running() override;


 // Modification (2025-06-01 07:48 PM -04): Added renderFrame method to update processed frame texture
    /**
     * @brief Updates the processed frame texture with YUYV data
     * @param dataPtr Pointer to YUYV data (from ZeroCopyFrameData::dataPtr)
     * @param width Frame width
     * @param height Frame height
     */
    void renderFrame_1(void* dataPtr, int width, int height);

    // ---------------- Additional Public Methods ----------------

    /**
     * @brief A static helper to convert YUYV to RGB
     * @param yuyv   Pointer to YUYV data
     * @param rgb    Output buffer for RGB24 data
     * @param width  Frame width
     * @param height Frame height
     */
    static void yuyvToRGB(const uint8_t* yuyv, uint8_t* rgb, int width, int height);

private:
    // ---------------- SDL Resource Cleanup ----------------
    struct SDL_Deleter {
        void operator()(SDL_Window* w)   const { if (w) SDL_DestroyWindow(w); }
        void operator()(SDL_Renderer* r) const { if (r) SDL_DestroyRenderer(r); }
        void operator()(SDL_Texture* t)  const { if (t) SDL_DestroyTexture(t); }
    };

    /**
     * @brief Logs or reports an error
     * @param msg The error message
     */
    void reportError(const std::string& msg);

    /**
     * @brief Creates the two SDL_Texture objects (for orig & proc) after renderer_ is created
     * @param width  Texture width
     * @param height Texture height
     */
    void initializeTextures(int width, int height);

    /**
     * @brief Internal event handler for SDL, sets running_=false on quit
     */
    void processEvents();

    /**
     * @brief Actually draws the frames on the renderer and calls SDL_RenderPresent
     */
    void renderFrame();


    /**
     * @brief Destroys all SDL resources (window, renderer, textures) 
     * 
     */

    void destroySDLResources();

    // -------------- Internal State --------------

    // SDL resources, managed by unique_ptr with custom deleters
    std::unique_ptr<SDL_Window, SDL_Deleter>   window_;
    std::unique_ptr<SDL_Renderer, SDL_Deleter> renderer_;
    std::unique_ptr<SDL_Texture, SDL_Deleter>  texOrig_;
    std::unique_ptr<SDL_Texture, SDL_Deleter>  texProc_;

    std::atomic<bool> running_{false};
    std::atomic<bool> initialized_{false};
    mutable std::mutex stateMutex_;

    DisplayConfig config_;                ///< Display configuration
    int displayWidth_{0};
    int displayHeight_{0};

    // Queues for original and processed frames
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> originalQueue_;
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> processedQueue_;
    // DataConcrete to support SystemMetricsAggregatorImpl injection and push camera metrics in real-time using the aggregator
    std::shared_ptr<ISystemMetricsAggregator> metricAggregator_;


    // Buffers for last frames (used if new frames not available)
    std::shared_ptr<ZeroCopyFrameData> lastOriginal_;
    std::shared_ptr<ZeroCopyFrameData> lastProcessed_;

    // Temporary buffers for YUYV->RGB conversion
    std::vector<uint8_t> origRgbBuffer_;
    std::vector<uint8_t> procRgbBuffer_;

    // Performance metrics
    std::chrono::steady_clock::time_point lastFrameTime_;
    double averageFrameTime_{0.0};
    int frameCounter_{0};
    std::chrono::steady_clock::time_point performanceCheckTime_;

    // Optional error callback
    std::function<void(const std::string&)> errorCallback_;


 
};

// ====================== Implementation ======================

inline SdlDisplayConcrete::SdlDisplayConcrete(
    // MODIFICATION: Added explicit deleter for dummy buffer
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> origQueue,
    std::shared_ptr<SharedQueue<std::shared_ptr<ZeroCopyFrameData>>> procQueue,
    std::shared_ptr<ISystemMetricsAggregator> aggregator
    // DataConcrete to support SystemMetricsAggregatorImpl injection and push camera metrics in real-time using the aggregator
)
  : originalQueue_(std::move(origQueue))
  , processedQueue_(std::move(procQueue))
  , metricAggregator_(std::move(aggregator))
  , lastFrameTime_(std::chrono::steady_clock::now())
  , performanceCheckTime_(std::chrono::steady_clock::now())
{
    // Initialize with valid dummy data (gray YUYV)
    // MODIFICATION: Corrected dummy buffer initialization to use shared_ptr<void>
    int dummyWidth = 640;
    int dummyHeight = 480;
    size_t dummySize = dummyWidth * dummyHeight * 2; // YUYV: 2 bytes per pixel
    
    auto dummyData = std::make_shared<std::vector<uint8_t>>(dummySize, 128); // Y=128, U=128, V=128 (gray)

    lastOriginal_ = std::make_shared<ZeroCopyFrameData>(
        dummyData,
        dummySize,
        dummyWidth,
        dummyHeight,
        0, // bufferIndex
        0  // frameNumber
    );
    lastProcessed_ = lastOriginal_;

    spdlog::debug("[SdlDisplayConcrete] Constructor complete.");
}

inline SdlDisplayConcrete::~SdlDisplayConcrete() {
    closeDisplay();
    spdlog::debug("[SdlDisplayConcrete] Destructor complete.");
}



// 1) In the private section add a helper:
void SdlDisplayConcrete::destroySDLResources()
{
    if(texOrig_)   { SDL_DestroyTexture(texOrig_.release());   }
    if(texProc_)   { SDL_DestroyTexture(texProc_.release());   }
    if(renderer_)  { SDL_DestroyRenderer(renderer_.release()); }
    if(window_)    { SDL_DestroyWindow(window_.release());     }
    SDL_Quit();                 // <- KEY: guarantees window disappears
}

inline void SdlDisplayConcrete::reportError(const std::string& msg) {
    if (errorCallback_) {
        errorCallback_(msg);
    } else {
        spdlog::error("[SdlDisplayConcrete] {}", msg);
    }
}

inline bool SdlDisplayConcrete::configure(const DisplayConfig& config) {
    std::lock_guard<std::mutex> lock(stateMutex_);

    if (config.width <= 0 || config.height <= 0) {
        reportError("Invalid display configuration (width/height <= 0).");
        return false;
    }
    config_ = config;
    spdlog::info("[SdlDisplayConcrete] Display configured: {}x{}, fullScreen={}",
                 config.width, config.height, config.fullScreen);
    return true;
}

inline bool SdlDisplayConcrete::initializeDisplay(int width, int height) {
    if (initialized_) {
        spdlog::warn("[SdlDisplayConcrete] Display is already initialized.");
        return true;
    }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        reportError(std::string("SDL_Init failed: ") + SDL_GetError());
        return false;
    }

    Uint32 windowFlags = SDL_WINDOW_SHOWN;
    if (config_.fullScreen) {
        windowFlags |= SDL_WINDOW_FULLSCREEN;
    }

    window_.reset(SDL_CreateWindow(
        config_.windowTitle.empty() ? "ERL Display" : config_.windowTitle.c_str(),
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width, height, windowFlags
    ));
    if (!window_) {
        reportError(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
        SDL_Quit();
        return false;
    }

    renderer_.reset(SDL_CreateRenderer(window_.get(), -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC));
    if (!renderer_) {
        reportError(std::string("SDL_CreateRenderer failed: ") + SDL_GetError());
        window_.reset();
        SDL_Quit();
        return false;
    }

    // Create textures for original & processed frames
    try {
        initializeTextures(width, height);
    } catch (const std::exception& e) {
        reportError(std::string("Failed to initialize textures: ") + e.what());
        window_.reset();
        renderer_.reset();
        SDL_Quit();
        return false;
    }

    // Mark as initialized
    initialized_   = true;
    running_       = true;
    displayWidth_  = width;
    displayHeight_ = height;

    spdlog::info("[SdlDisplayConcrete] SDL display initialized: {}x{}", width, height);
    return true;
}

inline void SdlDisplayConcrete::initializeTextures(int width, int height) {
    texOrig_.reset(SDL_CreateTexture(renderer_.get(),
        SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STREAMING,
        width, height
    ));
    texProc_.reset(SDL_CreateTexture(renderer_.get(),
        SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STREAMING,
        width, height
    ));

    if (!texOrig_ || !texProc_) {
        throw std::runtime_error("Failed to create SDL textures.");
    }
    spdlog::debug("[SdlDisplayConcrete] Textures created for {}x{} frames.", width, height);
}

inline void SdlDisplayConcrete::updateOriginalFrame(const uint8_t* rgbData, int width, [[maybe_unused]]int height) {
    std::lock_guard<std::mutex> lock(stateMutex_);

    if (!texOrig_ || !rgbData) return;

    // SDL_UpdateTexture expects pitch in bytes
    if (SDL_UpdateTexture(texOrig_.get(), nullptr, rgbData, width * 3) != 0) {
        reportError(std::string("Failed to update original texture: ") + SDL_GetError());
    }
}

inline void SdlDisplayConcrete::updateProcessedFrame(const uint8_t* rgbData, int width, [[maybe_unused]]int height) {
    std::lock_guard<std::mutex> lock(stateMutex_);

    if (!texProc_ || !rgbData) return;

    if (SDL_UpdateTexture(texProc_.get(), nullptr, rgbData, width * 3) != 0) {
        reportError(std::string("Failed to update processed texture: ") + SDL_GetError());
    }
}

inline bool SdlDisplayConcrete::is_Running() {
    return running_;
}

inline void SdlDisplayConcrete::renderAndPollEvents() {
    // MODIFIED: Change 'const int' to 'size_t'
    if (!running_) {
        spdlog::debug("[SdlDisplayConcrete] renderAndPollEvents called but not running.");
        return;
    }

    // 1. Process SDL events
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT ||
           (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)) {
            spdlog::info("[SdlDisplayConcrete] Quit/ESC event received; stopping display.");
            running_ = false;
            return;
        }
    }

    auto frameStartTime = std::chrono::steady_clock::now();

    // 2. Try popping new frames from originalQueue_ and processedQueue_
    std::shared_ptr<ZeroCopyFrameData> newOriginal, newProcessed;
    bool hasNewOriginal  = false;
    bool hasNewProcessed = false;

    if (originalQueue_) {
        hasNewOriginal = originalQueue_->try_pop(newOriginal);
    }
    if (processedQueue_) {
        hasNewProcessed = processedQueue_->try_pop(newProcessed);
    }

    // Decide which frames to use (fall back to last ones if none available)
    auto currentOriginal  = (hasNewOriginal && newOriginal) ? newOriginal : lastOriginal_;
    auto currentProcessed = (hasNewProcessed && newProcessed) ? newProcessed : lastProcessed_;

    // If valid new frames, store them as last
    if (hasNewOriginal && currentOriginal) {
        lastOriginal_ = currentOriginal;
    }
    if (hasNewProcessed && currentProcessed) {
        lastProcessed_ = currentProcessed;
    }

    // In SdlDisplayConcrete::renderAndPollEvents
    // 3a. Process Original Frame
    if (currentOriginal && currentOriginal->dataPtr) {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        // Calculate required buffer size FIRST
       // const int requiredSize = currentOriginal->width * currentOriginal->height * 3;
        // MODIFIED: Change 'const int' to 'size_t'
        const size_t requiredSize = currentOriginal->width * currentOriginal->height * 3;
        
        // Only resize if needed (avoids unnecessary reallocations)
        if (origRgbBuffer_.size() < requiredSize) {
            origRgbBuffer_.resize(requiredSize);
        }

        // Perform conversion with validated buffer size
        yuyvToRGB(
            static_cast<const uint8_t*>(currentOriginal->dataPtr),
            origRgbBuffer_.data(),
            currentOriginal->width,
            currentOriginal->height
        );

        // Update texture if valid
        if (texOrig_) {
            SDL_UpdateTexture(texOrig_.get(), nullptr, origRgbBuffer_.data(),
                            currentOriginal->width * 3);
        }
    }

    // 3b. Process Processed Frame
    if (currentProcessed && currentProcessed->dataPtr) {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        // Calculate required buffer size FIRST
        size_t requiredSize = currentProcessed->width * currentProcessed->height * 3;
        
        // Only resize if needed
        if (procRgbBuffer_.size() < requiredSize) {
            procRgbBuffer_.resize(requiredSize);
        }

        // Perform conversion with validated buffer size
        yuyvToRGB(
            static_cast<const uint8_t*>(currentProcessed->dataPtr),
            procRgbBuffer_.data(),
            currentProcessed->width,
            currentProcessed->height
        );

        // Update texture if valid
        if (texProc_) {
            SDL_UpdateTexture(texProc_.get(), nullptr, procRgbBuffer_.data(),
                            currentProcessed->width * 3);
        }
    }


    // 4. Render
    renderFrame();


    // 5. Simple performance measurement
    frameCounter_++;
    auto frameEndTime = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(frameEndTime - frameStartTime).count();

    // Calculate latency as the time taken to render the current frame
    double renderLatencyMs = elapsedMs;

    // Calculate dropped frames as the difference between expected frames and rendered frames
    static int totalFramesExpected = 0;
    totalFramesExpected++;
    int droppedFrames = totalFramesExpected - frameCounter_;

    /**
     *  Explanation:
        Latency: renderLatencyMs is calculated as the time between the start and end of the render process for the current frame.
        Dropped Frames: totalFramesExpected is incremented each time the render loop runs, assuming that a frame should be rendered each iteration. 
        droppedFrames is the difference between totalFramesExpected and frameCounter_, which counts how many frames have actually been rendered.
     * 
     */

    // Log FPS periodically
    if (frameCounter_ % 60 == 0) {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - performanceCheckTime_).count();
        double fps = (60.0 * 1000.0) / (double)duration;
        spdlog::info("[SdlDisplayConcrete] ~FPS: {:.1f}", fps);
        performanceCheckTime_ = now;
    }

    // After rendering, push metrics to aggregator
    if (metricAggregator_) {
        SystemMetricsSnapshot snapshot;
        snapshot.timestamp = std::chrono::system_clock::now();
        snapshot.displayStats.latencyMs = renderLatencyMs;
        snapshot.displayStats.droppedFrames = droppedFrames;
        metricAggregator_->pushMetrics(snapshot.timestamp, [&](SystemMetricsSnapshot& snap) {
            snap.displayStats = snapshot.displayStats;
        });
    }


    // 6. Throttle to ~60 FPS (simple approach)
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
}

// Modification (2025-06-01 07:48 PM -04): Added renderFrame method implementation
inline void SdlDisplayConcrete::renderFrame_1(void* dataPtr, int width, int height) {
    std::lock_guard<std::mutex> lock(stateMutex_);

    if (!running_ || !texProc_ || !dataPtr) {
        spdlog::warn("[SdlDisplayConcrete] renderFrame called but not running or invalid data/texProc_");
        return;
    }

    size_t requiredSize = width * height * 3;
    if (procRgbBuffer_.size() < requiredSize) {
        procRgbBuffer_.resize(requiredSize);
    }

    yuyvToRGB(
        static_cast<const uint8_t*>(dataPtr),
        procRgbBuffer_.data(),
        width,
        height
    );

    if (SDL_UpdateTexture(texProc_.get(), nullptr, procRgbBuffer_.data(), width * 3) != 0) {
        reportError(std::string("Failed to update processed texture in renderFrame: ") + SDL_GetError());
    }
}



inline void SdlDisplayConcrete::renderFrame() {
    if (!renderer_) {
        spdlog::error("[SdlDisplayConcrete] Renderer is null, cannot render.");
        return;
    }

    SDL_RenderClear(renderer_.get());

    // Draw original on left half
    if (texOrig_) {
        SDL_Rect dstLeft = {0, 0, displayWidth_ / 2, displayHeight_};
        SDL_RenderCopy(renderer_.get(), texOrig_.get(), nullptr, &dstLeft);
    }

    // Draw processed on right half
    if (texProc_) {
        SDL_Rect dstRight = {displayWidth_ / 2, 0, displayWidth_ / 2, displayHeight_};
        SDL_RenderCopy(renderer_.get(), texProc_.get(), nullptr, &dstRight);
    }

    SDL_RenderPresent(renderer_.get());
}

inline void SdlDisplayConcrete::setErrorCallback(std::function<void(const std::string&)> callback) {
    errorCallback_ = std::move(callback);
}

inline void SdlDisplayConcrete::closeDisplay() {
    running_ = false;
    destroySDLResources();
    // Reset unique_ptrs to ensure proper cleanup
    texOrig_.reset();
    texProc_.reset();
    renderer_.reset();
    window_.reset();
    SDL_Quit();
    initialized_ = false;
    spdlog::info("[SdlDisplayConcrete] Display closed.");
}

inline void SdlDisplayConcrete::yuyvToRGB(const uint8_t* yuyv, uint8_t* rgb, int width, int height) {
    auto clamp = [](int x) { return (x < 0) ? 0 : (x > 255 ? 255 : x); };
    int pixels = width * height;
    for (int i = 0, j = 0; i < pixels * 2; i += 4, j += 6) {
        // Y0 U Y1 V
        uint8_t Y0 = yuyv[i + 0];
        uint8_t U  = yuyv[i + 1];
        uint8_t Y1 = yuyv[i + 2];
        uint8_t V  = yuyv[i + 3];

        int C1 = (int)Y0 - 16;
        int C2 = (int)Y1 - 16;
        int D  = (int)U  - 128;
        int E  = (int)V  - 128;

        // Pixel 1
        int R1 = clamp(( 298 * C1           + 409 * E + 128) >> 8);
        int G1 = clamp(( 298 * C1 - 100 * D - 208 * E + 128) >> 8);
        int B1 = clamp(( 298 * C1 + 516 * D           + 128) >> 8);

        // Pixel 2
        int R2 = clamp(( 298 * C2           + 409 * E + 128) >> 8);
        int G2 = clamp(( 298 * C2 - 100 * D - 208 * E + 128) >> 8);
        int B2 = clamp(( 298 * C2 + 516 * D           + 128) >> 8);

        rgb[j + 0] = (uint8_t)R1;
        rgb[j + 1] = (uint8_t)G1;
        rgb[j + 2] = (uint8_t)B1;

        rgb[j + 3] = (uint8_t)R2;
        rgb[j + 4] = (uint8_t)G2;
        rgb[j + 5] = (uint8_t)B2;
    }
}

#endif // SDL_DISPLAY_CONCRETE_NEW_H



///=====================TESTING  code Final Version ==============================