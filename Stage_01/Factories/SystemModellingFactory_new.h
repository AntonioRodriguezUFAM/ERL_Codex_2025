// SystemModellingFactory_new.h
// (Stage 1 Factory: SoC, Data, Algorithm, Display Setup)


#ifndef SYSTEMMODELLINGFACTORY_NEW_H
#define SYSTEMMODELLINGFACTORY_NEW_H

#include <memory>
#include <vector>
#include <string>
#include <mutex>

// Interfaces
#include "../Interfaces/ISoC.h"
#include "../Interfaces/IData.h"
#include "../Interfaces/IAlgorithm.h"
#include "../Interfaces/IDisplay.h"

// Concrete classes
#include "../Concretes/SoCConcrete_new.h"
#include "../Concretes/AlgorithmConcrete_new.h"
#include "../Concretes/DataConcrete_new.h"
#include "../Concretes/SdlDisplayConcrete_new.h"

// Config structures
#include "../SharedStructures/SoCConfig.h"
#include "../SharedStructures/CameraConfig.h"
#include "../SharedStructures/AlgorithmConfig.h"
#include "../SharedStructures/DisplayConfig.h"

// SharedQueue & FrameData
#include "../SharedStructures/SharedQueue.h"
#include "../SharedStructures/FrameData.h"

// Central Thread Management
#include "../SharedStructures/ThreadManager.h"

/**
 * @class SystemModelingFactory
 * @brief Creates and configures the core system components (SoC, Data, Algorithm, Display)
 *        injecting shared queues for camera-to-algorithm and camera/algorithm-to-display pipelines.
 */
// class SystemModelingFactory {
// public:
//     SystemModelingFactory(const SoCConfig& socCfg,
//                           const CameraConfig& camCfg,
//                           const AlgorithmConfig& algoCfg,
//                           const DisplayConfig& dispCfg)
//         : socConfig_(socCfg),
//           camConfig_(camCfg),
//           algoConfig_(algoCfg),
//           dispConfig_(dispCfg)
//     {
//         // Create shared queues for pipeline
//          // Create the shared queues only once
//         algoQueue_        = std::make_shared<SharedQueue<FrameData>>(5);
//         displayOrigQueue_ = std::make_shared<SharedQueue<FrameData>>(5);
//         processedQueue_   = std::make_shared<SharedQueue<FrameData>>(5);

//         // Construct components
//           // Construct components and inject the same queues into each
//         soc_     = std::make_unique<SoCConcrete>();
//         data_    = std::make_unique<DataConcrete>(algoQueue_, displayOrigQueue_);
//         algo_    = std::make_unique<AlgorithmConcrete>(algoQueue_, processedQueue_, threadManager_);
//         display_ = std::make_unique<SdlDisplayConcrete>(displayOrigQueue_, processedQueue_);
//     }

//     // Apply config to each component
//     void configureSystemModel() {
//         if (soc_)     soc_->configure(socConfig_);
//         if (data_)    data_->configure(camConfig_);
//         if (algo_)    algo_->configure(algoConfig_);
//         if (display_) display_->configure(dispConfig_);
//     }

//     // Accessors
//     ISoC*        getSoC()       const { return soc_.get(); }
//     IData*       getData()      const { return data_.get(); }
//     IAlgorithm*  getAlgorithm() const { return algo_.get(); }
//     IDisplay*    getDisplay()   const { return display_.get(); }

//     // SharedQueues
//     // Accessors to allow other parts of the system to use the same queues
//     std::shared_ptr<SharedQueue<FrameData>> getAlgoQueue()        const { return algoQueue_; }
//     std::shared_ptr<SharedQueue<FrameData>> getDisplayOrigQueue() const { return displayOrigQueue_; }
//     std::shared_ptr<SharedQueue<FrameData>> getProcessedQueue()   const { return processedQueue_; }


//      // ... other methods ...

// private:
//     // Configuration
//     SoCConfig       socConfig_;
//     CameraConfig    camConfig_;
//     AlgorithmConfig algoConfig_;
//     DisplayConfig   dispConfig_;

//     // Concrete components
//     std::unique_ptr<SoCConcrete>         soc_;
//     std::unique_ptr<DataConcrete>        data_;
//     std::unique_ptr<AlgorithmConcrete>   algo_;
//     std::unique_ptr<IDisplay>           display_;

//     // SharedQueues for pipeline
//     std::shared_ptr<SharedQueue<FrameData>> algoQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> processedQueue_;

//     // Thread Manager instance
//     ThreadManager threadManager_;
// };



//==================================

// SystemModelingFactory_new.h
class SystemModelingFactory {
public:
    SystemModelingFactory(const SoCConfig& socCfg,
                          const CameraConfig& camCfg,
                          const AlgorithmConfig& algoCfg,
                          const DisplayConfig& dispCfg)
        : socConfig_(socCfg),
          camConfig_(camCfg),
          algoConfig_(algoCfg),
          dispConfig_(dispCfg)
    {
        // Create shared queues (only once)
        algoQueue_        = std::make_shared<SharedQueue<FrameData>>(5);
        displayOrigQueue_ = std::make_shared<SharedQueue<FrameData>>(5);
        processedQueue_   = std::make_shared<SharedQueue<FrameData>>(5);

        // Create components as shared_ptr
        soc_     = std::make_shared<SoCConcrete>();
        data_    = std::make_shared<DataConcrete>(algoQueue_, displayOrigQueue_);
        algo_    = std::make_shared<AlgorithmConcrete>(algoQueue_, processedQueue_, threadManager_);

        // Create the display instance using your SDL-based display concrete
        display_ = std::make_shared<SdlDisplayConcrete>(displayOrigQueue_, processedQueue_);

        // Configure display with dispCfg
        // Configure display immediately
        display_->configure(dispCfg);
        
    }

    // Getters now return shared_ptr
    std::shared_ptr<ISoC> getSoC() const { return soc_; }
    std::shared_ptr<IData> getData() const { return data_; }
    std::shared_ptr<IAlgorithm> getAlgorithm() const { return algo_; }
    std::shared_ptr<IDisplay> getDisplay() const { return display_; }

    // Also provide getters for the queues
    std::shared_ptr<SharedQueue<FrameData>> getAlgoQueue() const { return algoQueue_; }
    std::shared_ptr<SharedQueue<FrameData>> getDisplayOrigQueue() const { return displayOrigQueue_; }
    std::shared_ptr<SharedQueue<FrameData>> getProcessedQueue() const { return processedQueue_; }

private:
    SoCConfig       socConfig_;
    CameraConfig    camConfig_;
    AlgorithmConfig algoConfig_;
    DisplayConfig   dispConfig_;

    std::shared_ptr<SoCConcrete> soc_;
    std::shared_ptr<DataConcrete> data_;
    std::shared_ptr<AlgorithmConcrete> algo_;
    std::shared_ptr<IDisplay> display_;

    std::shared_ptr<SharedQueue<FrameData>> algoQueue_;
    std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue_;
    std::shared_ptr<SharedQueue<FrameData>> processedQueue_;

    ThreadManager threadManager_;
};

#endif // SYSTEMMODELLINGFACTORY_NEW_H
