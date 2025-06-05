// ZeroCopyFrameData.h

// ZeroCopyFrameData.h (if you keep it in a separate file)
// #pragma once
// #include <cstdint>
// #include <chrono>

// struct ZeroCopyFrameData {
//     std::shared_ptr<void> bufferData; // Shared ownership of the buffer
//     uint8_t* dataPtr;                // Pointer to buffer data
//    // void*  dataPtr;      // pointer to mapped buffer
//    // ... other members (width, height, etc.)
//     size_t size;         // total bytes used in the buffer
//     int    width;        
//     int    height;
//     size_t bufferIndex;
//     int    frameNumber;
//     // optional timestamp:
//     std::chrono::steady_clock::time_point captureTime;  // Capture timestamp
// };


// #pragma once
// #include <cstdint>
// #include <memory>
// #include <chrono>
// // In SharedStructures/ZeroCopyFrameData.h
// class ZeroCopyFrameData {
//     public:
//         ZeroCopyFrameData(
//             std::shared_ptr<std::vector<uint8_t>> data,
//             size_t size,
//             int width,
//             int height,
//             int bufferIndex,
//             int frameNumber
//         ) : dataHolder(data),
//             dataPtr(data ? data->data() : nullptr),
//             size(size),
//             width(width),
//             height(height),
//             bufferIndex(bufferIndex),
//             frameNumber(frameNumber) {}
    
//         void* dataPtr;
//         size_t size;
//         int width;
//         int height;
//         int bufferIndex;
//         int frameNumber;
    
//     private:
//         std::shared_ptr<std::vector<uint8_t>> dataHolder; // Keeps the vector alive
//     };


#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <chrono>

class ZeroCopyFrameData {
public:
    // Add default constructor (if allowed by your logic)
    ZeroCopyFrameData() = default; 

    // Existing parameterized constructor
    // Modified initializer list to match member declaration order
    ZeroCopyFrameData(
        std::shared_ptr<std::vector<uint8_t>> data,
        size_t size,
        int width,
        int height,
        int bufferIndex,
        int frameNumber
    ) : 
        dataPtr(data ? data->data() : nullptr),  // Matches 1st declaration
        size(size),                              // Matches 2nd
        width(width),                            // Matches 3rd
        height(height),                          // Matches 4th
        bufferIndex(bufferIndex),                // Matches 5th
        frameNumber(frameNumber),                // Matches 6th
        dataHolder(data)                         // Matches 7th (last)
    {}

    // Members declared in the same order as initialization
    void* dataPtr;
    size_t size;
    int width;
    int height;
    int bufferIndex;
    int frameNumber;

private:
    std::shared_ptr<std::vector<uint8_t>> dataHolder; 
};