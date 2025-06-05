//SystemModellingFactory_new.h
#ifndef SYSTEMMODELLINGFACTORY_NEW_H
#define SYSTEMMODELLINGFACTORY_NEW_H

#include <memory>
#include <vector>
#include <string>

// Include interfaces
#include "../Interfaces/IRawDataFormat.h"
#include "../Interfaces/IData.h"
#include "../Interfaces/IAlgorithm.h"
#include "../Interfaces/ISoC.h"
#include "../Interfaces/IConstraints.h"
#include "../Interfaces/IOptimization.h"
#include "../Interfaces/ILearning.h"
#include "../Interfaces/IPerformance.h"
#include "../Interfaces/IDisplay.h"

// Include concrete classes
#include "../Concretes/RawDataFormatConcrete.h"
#include "../Concretes/SoCConcrete.h"
#include "../Concretes/AlgorithmConcrete_new.h" // Updated version
#include "../Concretes/DataConcrete_new.h"        // Updated version
#include "../Concretes/ConstraintsConcrete.h"
#include "../Concretes/OptimizationConcrete.h"
#include "../Concretes/LearningConcrete.h"
#include "../Concretes/SdlDisplayConcrete_new.h"  // Updated version

// Include configuration structures
#include "../SharedStructures/SoCConfig.h"
#include "../SharedStructures/CameraConfig.h"
#include "../SharedStructures/AlgorithmConfig.h"
#include "../SharedStructures/DisplayConfig.h"

// Include SharedQueue and FrameData
#include "../SharedStructures/SharedQueue.h"
#include "../SharedStructures/FrameData.h"
#include "../SharedStructures/ThreadManager.h" // ✅ Added centralized thread management

/**
 * @class SystemModelingFactory
 * @brief Creates and configures the core system components (SoC, Data, Algorithm, Display)
 *        using dependency injection of shared queues.
 *
 * This factory creates three SharedQueue objects:
 *   - algoQueue: For frames from DataConcrete to AlgorithmConcrete.
 *   - displayOrigQueue: For original frames from DataConcrete to SdlDisplayConcrete.
 *   - processedQueue: For processed frames from AlgorithmConcrete to SdlDisplayConcrete.
 *
 * These queues are managed by shared_ptr so that their lifetimes are maintained.
 */
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
        // Create the shared queues (the lifetime of these queues will now be managed via shared_ptr)
        algoQueue_ = std::make_shared<SharedQueue<FrameData>>();
        displayOrigQueue_ = std::make_shared<SharedQueue<FrameData>>();
        processedQueue_ = std::make_shared<SharedQueue<FrameData>>();

        // Create the components and inject the shared queues.
        // (Your updated DataConcrete_new.h must define a constructor that accepts
        // std::shared_ptr<SharedQueue<FrameData>> for the algorithm and display queues.)
        soc_ = std::make_unique<SoCConcrete>();
        data_ = std::make_unique<DataConcrete>(algoQueue_, displayOrigQueue_);
//        algo_ = std::make_unique<AlgorithmConcrete>(algoQueue_, processedQueue_);
        algo_ = std::make_unique<AlgorithmConcrete>(algoQueue_, processedQueue_, threadManager_);

        // New code:
      //  algo_ = std::make_unique<AlgorithmConcrete>(algoQueue, processedQueue, threadManager_);

        display_ = std::make_unique<SdlDisplayConcrete>(displayOrigQueue_, processedQueue_);
    }

    // Call this method to apply configuration settings to each component.
    void configureSystemModel() {
        if (soc_) {
            soc_->configure(socConfig_);
        }
        if (data_) {
            data_->configure(camConfig_);
        }
        if (algo_) {
            algo_->configure(algoConfig_);
        }
        if (display_) {
            display_->configure(dispConfig_);
        }
    }

    // Accessors for each component.
    ISoC* getSoC() const { return soc_.get(); }
    IData* getData() const { return data_.get(); }
    IAlgorithm* getAlgorithm() const { return algo_.get(); }
    IDisplay* getDisplay() const { return display_.get(); }

    // Optionally, getters for the shared queues.
    std::shared_ptr<SharedQueue<FrameData>> getAlgoQueue() const { return algoQueue_; }
    std::shared_ptr<SharedQueue<FrameData>> getDisplayOrigQueue() const { return displayOrigQueue_; }
    std::shared_ptr<SharedQueue<FrameData>> getProcessedQueue() const { return processedQueue_; }

private:
    // Configuration objects.
    SoCConfig socConfig_;
    CameraConfig camConfig_;
    AlgorithmConfig algoConfig_;
    DisplayConfig dispConfig_;

    // Core components.
    std::unique_ptr<SoCConcrete> soc_;
    std::unique_ptr<DataConcrete> data_;
    std::unique_ptr<AlgorithmConcrete> algo_;
    std::unique_ptr<IDisplay> display_;

    // SharedQueue objects.
    std::shared_ptr<SharedQueue<FrameData>> algoQueue_;
    std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue_;
    std::shared_ptr<SharedQueue<FrameData>> processedQueue_;

    // Ensure ThreadManager instance exists
    ThreadManager threadManager_;
};

#endif // SYSTEMMODELLINGFACTORY_NEW_H



// // SystemModellingFactory_new.H

// /**
//  * 
//  * SystemModelingFactory with Dependency Injection and Configuration
//  * This updated version of SystemModelingFactory demonstrates dependency injection
//  * for each core component (ISoC, IData, IAlgorithm) as well as their configuration objects 
//  * (SoCConfig, CameraConfig, AlgorithmConfig, and optionally DisplayConfig). 
//  * By injecting dependencies and configurations, 
//  * we decouple the factory from specific concrete classes and make it easier to test, extend, 
//  * or swap different implementations.
//  * 
//  */

// // SystemModellingFactory_new.h
// #pragma once
// #include <memory>
// #include <vector>
// #include <string>

// // Include interfaces
// #include "../Interfaces/IRawDataFormat.h"
// #include "../Interfaces/IData.h"
// #include "../Interfaces/IAlgorithm.h"
// #include "../Interfaces/ISoC.h"
// #include "../Interfaces/IConstraints.h"
// #include "../Interfaces/IOptimization.h"
// #include "../Interfaces/ILearning.h"
// #include "../Interfaces/IPerformance.h"
// #include "../Interfaces/IDisplay.h"

// // Include concrete classes
// #include "../Concretes/RawDataFormatConcrete.h"
// #include "../Concretes/SoCConcrete.h"
// #include "../Concretes/AlgorithmConcrete_new.h"
// #include "../Concretes/DataConcrete_new.h"
// #include "../Concretes/ConstraintsConcrete.h"
// #include "../Concretes/OptimizationConcrete.h"
// #include "../Concretes/LearningConcrete.h"
// #include "../Concretes/SdlDisplayConcrete_new.h"

// // Include configuration structures
// #include "../SharedStructures/SoCConfig.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/AlgorithmConfig.h"
// #include "../SharedStructures/DisplayConfig.h"

// // Include our SharedQueue
// #include "../SharedStructures/SharedQueue.h"
// #include "../SharedStructures/FrameData.h"


// /**
//  * @class SystemModelingFactory
//  * @brief Creates and configures the core system components by injecting
//  *        SharedQueue instances for inter-task communication.
//  */
// class SystemModelingFactory
// {
// public:
//     /**
//      * @brief Constructor with dependency injection for SoC, Data, Algorithm, 
//      *        plus configuration for each component. 
//      * 
//      * @param soc         A unique_ptr to an ISoC implementation
//      * @param data        A unique_ptr to an IData (camera/data source) implementation
//      * @param algo        A unique_ptr to an IAlgorithm implementation
//      * @param display     A unique_ptr to an IDisplay implementation
//      * @param socConfig   Configuration parameters for SoC monitoring
//      * @param cameraConfig Configuration parameters for the data/camera
//      * @param algoConfig  Configuration parameters for the algorithm
//      * @param displayConfig (optional) Configuration parameters for the display 
//      */
//      SystemModelingFactory(
//         std::unique_ptr<ISoC> soc,
//         const SoCConfig& socCfg,
//         const CameraConfig& camCfg,
//         const AlgorithmConfig& algoCfg,
//         const DisplayConfig& dispCfg
//     ) : soc_(std::move(soc)),
//         socConfig_(socCfg),
//         camConfig_(camCfg),
//         algoConfig_(algoCfg),
//         dispConfig_(dispCfg)
//     {
//         // Create the shared queues:
//         algoQueue_ = std::make_shared<SharedQueue<FrameData>>();
//         displayOrigQueue_ = std::make_shared<SharedQueue<FrameData>>();
//         processedQueue_ = std::make_shared<SharedQueue<FrameData>>();

//         // Create components using dependency injection:
//         data_ = std::make_unique<DataConcrete>(*algoQueue_, *displayOrigQueue_);
//         algo_ = std::make_unique<AlgorithmConcrete>(*algoQueue_, *processedQueue_);
//         display_ = std::make_unique<SdlDisplayConcrete>(*displayOrigQueue_, *processedQueue_);
//     }


//     /**   // Optionally inject a display
//      * @brief Inject an optional display component separately (if your pipeline needs a display).
//      *        This method can be called before `initializeSystemModel()`.
//      * @param display A unique_ptr to an IDisplay implementation
//      */
//     void setDisplay(std::unique_ptr<IDisplay> display)
//     {
//         display_ = std::move(display);
//     }

//     /**
//      * @brief Initializes/configures each component with the injected configuration objects.
//      */
//        void configureSystemModel() {
//         if (soc_) {
//             soc_->configure(socConfig_);
//         }
//         if (data_) {
//             data_->configure(camConfig_);
//         }
//         if (algo_) {
//             algo_->configure(algoConfig_);
//         }
//         if (display_) {
//             display_->configure(dispConfig_);
//         }
//     }

//       // Accessor methods for each component (if needed)
//     ISoC* getSoC() const { return soc_.get(); }
//     IData* getData() const { return data_.get(); }
//     IAlgorithm* getAlgorithm() const { return algo_.get(); }
//     IDisplay* getDisplay() const { return display_.get(); }

// private:

//  // Dependencies
//     std::unique_ptr<ISoC> soc_;
//     std::unique_ptr<IData> data_;
//     std::unique_ptr<IAlgorithm> algo_;
//     std::unique_ptr<IDisplay> display_;

//     // Configuration for each dependency
//     SoCConfig socConfig_;
//     CameraConfig camConfig_;
//     AlgorithmConfig algoConfig_;
//     DisplayConfig dispConfig_;

//     // SharedQueue objects for inter-task communication:
//     std::shared_ptr<SharedQueue<FrameData>> algoQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> displayOrigQueue_;
//     std::shared_ptr<SharedQueue<FrameData>> processedQueue_;
// };
