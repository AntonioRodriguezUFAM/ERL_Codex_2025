// SharedQueue.h
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>


/* buffer pool 
* BufferPool class is a simple buffer pool implementation that provides a pool of buffers of a fixed size.
*Buffer Pooling with SharedQueue
*   Implementation Steps:
*   Create a buffer pool to manage reusable buffers
*   Acquire buffers from pool when needed
*   Return buffers to pool after processing
*/
class BufferPool {
    public:
        BufferPool(size_t bufferSize, size_t poolSize) 
            : bufferSize_(bufferSize), poolSize_(poolSize) {
            assert(bufferSize > 0 && poolSize > 0 && "Invalid buffer or pool size");
            for (size_t i = 0; i < poolSize; ++i) {
                void* buffer = malloc(bufferSize);
                if (!buffer) {
                spdlog::error("BufferPool: malloc failed for buffer size {}", bufferSize);
                throw std::bad_alloc();
                }
                freeBuffers_.push(buffer);
            }
        }
    
         void* acquireBuffer() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (freeBuffers_.empty()) {
            spdlog::warn("BufferPool: Expanding pool, allocating buffer of size {}", bufferSize_);
            void* buffer = malloc(bufferSize_);
            if (!buffer) {
                spdlog::error("BufferPool: malloc failed for buffer size {}", bufferSize_);
                throw std::bad_alloc();
            }
            return buffer;
        }
        void* buffer = freeBuffers_.front();
        freeBuffers_.pop();
        return buffer;
         }
    
     void releaseBuffer(void* buffer) {
        if (!buffer) {
            spdlog::warn("BufferPool: Attempt to release null buffer");
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        freeBuffers_.push(buffer);
    }
    
    private:
        size_t bufferSize_;
        size_t poolSize_;
        std::queue<void*> freeBuffers_;
        std::mutex mutex_;
    };
    


/**
 * @brief SharedQueue is a thread-safe queue implementation to support producer-consumer scenarios.
 * 
 * This class provides mechanisms for producers to push items into the queue and consumers to pop
 * items from it. The queue blocks consumers when it is empty and unblocks them when items are added
 * or the queue is stopped.
 * 
 * Optionally, you can enable "bounded capacity" by setting maxSize > 0.
 */
template <typename T>   // SharedQueue.h (no changes needed for template)
// Existing implementation handles T as shared_ptr<ZeroCopyFrameData>
class SharedQueue {
    
public:
    /**
     * @brief Construct a new SharedQueue with optional bounded capacity.
     * @param maxSize If > 0, queue is bounded. If 0, unbounded.
     * @param bufferPool Optional BufferPool for managing T's buffers.
     */
    // explicit SharedQueue(size_t maxSize = 10, std::shared_ptr<BufferPool> bufferPool = nullptr)
    //     : stop_(false), maxSize_(maxSize), bufferPool_(bufferPool) {
    //     assert(maxSize_ >= 0 && "Invalid maxSize");
    // }
    explicit SharedQueue(size_t maxSize, std::shared_ptr<BufferPool> bufferPool = nullptr)
    : maxSize_(maxSize), bufferPool_(bufferPool) {
    assert(maxSize_ > 0 && "Invalid maxSize");
    //queue_.reserve(maxSize_);
}
    /**
     * @brief Push an item into the queue (Producer).
     * Blocks if queue is full (bounded mode) until space is available or stop() is called.
     */
    void push(const T& item) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condNotFull_.wait(lock, [this] {
                return stop_ || (maxSize_ == 0) || (queue_.size() < maxSize_);
            });
            if (stop_) {
                spdlog::warn("SharedQueue: Push to stopped queue");
                return;
            }
            queue_.push(item);
        }
        condNotEmpty_.notify_one();
    }

    /**
     * @brief Push an rvalue reference (move).
     */
    void push(T&& item) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condNotFull_.wait(lock, [this] {
                return stop_ || (maxSize_ == 0) || (queue_.size() < maxSize_);
            });
            if (stop_) {
                 spdlog::warn("SharedQueue: Push to stopped queue");
                return;
            }
            queue_.push(std::move(item));
        }
        condNotEmpty_.notify_one();
    }

    /**
     * @brief Pop an item from the queue (Consumer). 
     * Blocks if queue is empty (and not stopped).
     * @return false if the queue is stopped and empty.
     * 
     * @attention:
     * The consumer will wait until the queue is not empty or stop() is called.
     * The SharedQueue class provides the methods you need to retrieve frames:
     * - bool pop(FrameData& item) (Blocking Pop)
     * - bool tryPop(FrameData& item) (Non-blocking Pop)
     * pop(FrameData& item) (Blocking Pop):
     * This method will block (wait) if the queue is currently empty until a new FrameData becomes available in the queue.
     * Once a FrameData is available, it is removed from the queue and copied into the item variable that you provide as an argument.
     * Use Case: Suitable for consumer threads that must process every frame and can afford to wait. If no frames are available, the thread will pause, saving CPU cycles.
     */
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condNotEmpty_.wait(lock, [this] {
            return stop_ || !queue_.empty();
        });
        if (stop_ && queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        if (maxSize_ > 0) {
            condNotFull_.notify_one();
        }
        return true;
    }



     /**
     * @brief try_pop method in the SharedQueue class. This method will attempt to pop an item from the queue without blocking
     * Try to pop an item from the queue without blocking.
     * @return false if the queue is stopped and empty.
     * 
     * @attention: try_pop(FrameData& item) (Non-Blocking Pop):
     * This method will not block. It immediately checks if there is a FrameData in the queue.
     * If a FrameData is available, it is removed from the queue and copied into item, and the method returns true.
     * If the queue is empty, it returns false immediately without waiting.
     * Use Case: Useful when the consumer thread needs to do other things if no frame is immediately available, 
     * or when you want to avoid blocking in certain scenarios. 
     * You would typically use this in a loop and check the return value to see if a frame was retriev
     */
    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false; // Return false if the queue is empty
        }
        item = std::move(queue_.front());
        queue_.pop();
        if (maxSize_ > 0) {
            condNotFull_.notify_one();
        }
        return true; // Return true if an item was popped
    }

/**
 * @brief is full method in the SharedQueue class. This method checks if the queue is full.
 * 
 */
    bool isFull() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return (maxSize_ > 0) && (queue_.size() >= maxSize_);
        //return queue_.size() >= maxSize_;
    }
    /**
     * @brief Stop the queue and unblock all producers/consumers.
     */
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        condNotEmpty_.notify_all();
        condNotFull_.notify_all();
    }

    /**
     * @brief Alias for stop().
     */
    void close() { stop(); }

    /**
     * @brief Return snapshot of current size (not always accurate due to concurrency).
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    size_t capacity() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return maxSize_;
    }

    /**
     * @brief Check whether stop() has been called.
     */
    bool isStopped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stop_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable condNotEmpty_;
    std::condition_variable condNotFull_;
    std::queue<T> queue_;
    bool stop_;
    size_t maxSize_;
    std::shared_ptr<BufferPool> bufferPool_;
};
