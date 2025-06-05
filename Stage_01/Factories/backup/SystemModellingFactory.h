
//SystemModellingFactory.h

#pragma once

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

// Include your concrete classes
#include "../Concretes/RawDataFormatConcrete.h"
#include "../Concretes/SoCConcrete.h"
#include "../Concretes/AlgorithmConcrete.h"
#include "../Concretes/DataConcrete.h"
#include "../Concretes/ConstraintsConcrete.h"
#include "../Concretes/OptimizationConcrete.h"
#include "../Concretes/LearningConcrete.h"
#include "../Concretes/SdlDisplayConcrete.h"

// Optional: performance logger
#include "../../Stage_02/Logger/PerformanceLogger.h"

/**
 * @class SystemModelingFactory
 * @brief Creates "concrete" instances (DataConcrete, AlgorithmConcrete, SoCConcrete, SdlDisplayConcrete, etc.)
 * 
 * In a larger system, this could read from config or dynamically detect hardware.
 */
class SystemModelingFactory {
public:
    SystemModelingFactory() = default;
    virtual ~SystemModelingFactory() = default;

    // Core
    virtual std::unique_ptr<IData>        createDataComponent();
    virtual std::unique_ptr<IAlgorithm>   createAlgorithmComponent();
    virtual std::unique_ptr<ISoC>         createSoCComponent();
    
    // Additional
    virtual std::unique_ptr<IRawDataFormat> createRawDataFormatComponent();
    virtual std::unique_ptr<IConstraints>   createConstraintsComponent();
    virtual std::unique_ptr<IOptimization>  createOptimizationComponent();
    virtual std::unique_ptr<ILearning>      createLearningComponent();
    virtual std::unique_ptr<IPerformance>   createPerformanceComponent();
    virtual std::unique_ptr<IDisplay>       createDisplayComponent();
};

// Implementation
inline std::unique_ptr<IData> SystemModelingFactory::createDataComponent() {
    return std::make_unique<DataConcrete>();
}

inline std::unique_ptr<IAlgorithm> SystemModelingFactory::createAlgorithmComponent() {
    return std::make_unique<AlgorithmConcrete>();
}

inline std::unique_ptr<ISoC> SystemModelingFactory::createSoCComponent() {
    return std::make_unique<SoCConcrete>();
}

inline std::unique_ptr<IDisplay> SystemModelingFactory::createDisplayComponent() {
    return std::make_unique<SdlDisplayConcrete>();
}

inline std::unique_ptr<IRawDataFormat> SystemModelingFactory::createRawDataFormatComponent() {
    return std::make_unique<RawDataFormatConcrete>();
}

inline std::unique_ptr<IConstraints> SystemModelingFactory::createConstraintsComponent() {
    return std::make_unique<ConstraintsConcrete>();
}

inline std::unique_ptr<IOptimization> SystemModelingFactory::createOptimizationComponent() {
    return std::make_unique<OptimizationConcrete>();
}

inline std::unique_ptr<ILearning> SystemModelingFactory::createLearningComponent() {
    return std::make_unique<LearningConcrete>();
}

inline std::unique_ptr<IPerformance> SystemModelingFactory::createPerformanceComponent() {
    return std::make_unique<PerformanceLogger>();
}




// /**
//  * @brief A factory class that creates "concrete" instances of IData, IAlgorithm, and ISoC.
//  * 
//  * In larger systems, this could read config files or detect hardware dynamically.
//  * For now, we simply instantiate DataConcrete, AlgorithmConcrete, and SoCConcrete.
//  * 
//  * Explanation
//  * 1. SystemModelingFactory simply returns new instances of your “concrete” classes.
//  * 2. If you want different camera types or algorithms, you can implement branching logic based on some config or environment.
//  */

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

// // Include shared structures
// #include "../SharedStructures/FrameData.h"
// #include "../SharedStructures/CameraConfig.h"
// #include "../SharedStructures/AlgorithmConfig.h"
// #include "../SharedStructures/SoCConfig.h"

// // Include concrete classes
// #include "../Concretes/RawDataFormatConcrete.h"
// #include "../Concretes/SoCConcrete.h"
// #include "../Concretes/AlgorithmConcrete.h"
// #include "../Concretes/DataConcrete.h"
// #include "../Concretes/ConstraintsConcrete.h"
// #include "../Concretes/OptimizationConcrete.h"
// #include "../Concretes/LearningConcrete.h"
// #include "../Concretes/SdlDisplayConcrete.h"
// #include "../../Stage_02/Logger/PerformanceLogger.h"

// /**
//  * @class SystemModelingFactory
//  * @brief A factory class that creates "concrete" instances of IData (cameras), 
//  *        IAlgorithm, ISoC, and other optional modules (IRawDataFormat, IConstraints, etc.).
//  * 
//  * In a larger system, this class could read config files or detect hardware dynamically.
//  * For now, it simply instantiates various Concrete classes (e.g., DataConcrete, AlgorithmConcrete, SoCConcrete).
//  *
//  * Usage:
//  *   - createDataComponent(): returns a DataConcrete capable of asynchronous streaming.
//  *   - createAlgorithmComponent(): returns an AlgorithmConcrete for processing frames.
//  *   - createSoCComponent(): returns a SoCConcrete for SoC monitoring.
//  *   - create*() for other modules if needed.
//  *
//  * The user or a higher-level coordinator (e.g., SystemCaptureFactory) can then wire up these components
//  * into an end-to-end pipeline (cameras -> processing -> display -> logging).
//  */
// class SystemModelingFactory {
// public:
//     SystemModelingFactory() = default;
//     virtual ~SystemModelingFactory() = default;

//     // ------------------------------------------------------------
//     // Core creation methods
//     // ------------------------------------------------------------

//     /**
//      * @brief Creates and returns a new IData (camera) instance.
//      * @return A unique_ptr to a DataConcrete object (async-capable camera).
//      */
//     virtual std::unique_ptr<IData> createDataComponent();

//     /**
//      * @brief Creates and returns a new IAlgorithm instance.
//      * @return A unique_ptr to an AlgorithmConcrete object.
//      */
//     virtual std::unique_ptr<IAlgorithm> createAlgorithmComponent();

//     /**
//      * @brief Creates and returns a new ISoC instance.
//      * @return A unique_ptr to a SoCConcrete object for SoC monitoring.
//      */
//     virtual std::unique_ptr<ISoC> createSoCComponent();

//     // ------------------------------------------------------------
//     // Additional modules creation
//     // ------------------------------------------------------------

//     /**
//      * @brief Creates and returns a new IRawDataFormat instance.
//      * @return A unique_ptr to a RawDataFormatConcrete object.
//      */
//     virtual std::unique_ptr<IRawDataFormat> createRawDataFormatComponent();

//     /**
//      * @brief Creates and returns a new IConstraints instance.
//      * @return A unique_ptr to a ConstraintsConcrete object.
//      */
//     virtual std::unique_ptr<IConstraints> createConstraintsComponent();

//     /**
//      * @brief Creates and returns a new IOptimization instance.
//      * @return A unique_ptr to an OptimizationConcrete object.
//      */
//     virtual std::unique_ptr<IOptimization> createOptimizationComponent();

//     /**
//      * @brief Creates and returns a new ILearning instance.
//      * @return A unique_ptr to a LearningConcrete object.
//      */
//     virtual std::unique_ptr<ILearning> createLearningComponent();

//     /**
//      * @brief Creates and returns a new IPerformance instance.
//      *        For example, PerformanceLogger to track performance metrics.
//      */
//     virtual std::unique_ptr<IPerformance> createPerformanceComponent();

//     /**
//      * @brief Creates and returns a new IDisplay instance.
//      * @return A unique_ptr to an SdlDisplayConcrete object.
//      */
//     virtual std::unique_ptr<IDisplay> createDisplayComponent();

//     // ------------------------------------------------------------
//     // Optionally, you could add specialized methods to create multiple cameras or multiple algorithms
//     // e.g., createDataComponents(size_t cameraCount), createAlgorithmComponents(size_t algoCount), etc.
//     // ------------------------------------------------------------
// };

// //===========================================================
// // Implementation of SystemModelingFactory
// //===========================================================
// inline std::unique_ptr<IRawDataFormat> SystemModelingFactory::createRawDataFormatComponent() {
//     return std::make_unique<RawDataFormatConcrete>();
// }

// inline std::unique_ptr<IData> SystemModelingFactory::createDataComponent() {
//     // Instantiate a new DataConcrete, which is typically asynchronous-capable if you spawn a thread for streaming.
//     auto camera = std::make_unique<DataConcrete>();

//     // Optionally, set a default error callback or apply default config if desired:
//     // camera->setErrorCallback([](const std::string& errMsg) {
//     //     spdlog::error("[DataConcrete-Default] {}", errMsg);
//     // });
//     // camera->configure(defaultCameraConfig);

//     return camera;
// }

// inline std::unique_ptr<IAlgorithm> SystemModelingFactory::createAlgorithmComponent() {
//     auto alg = std::make_unique<AlgorithmConcrete>();

//     // Optionally set error callbacks or default configs:
//     // alg->setErrorCallback(...);
//     // alg->configure(defaultAlgorithmConfig);

//     return alg;
// }

// inline std::unique_ptr<ISoC> SystemModelingFactory::createSoCComponent() {
//     auto soc = std::make_unique<SoCConcrete>();
//     // e.g., soc->configure(defaultSoCConfig);
//     return soc;
// }

// inline std::unique_ptr<IDisplay> SystemModelingFactory::createDisplayComponent() {
//     return std::make_unique<SdlDisplayConcrete>();
// }

// inline std::unique_ptr<IConstraints> SystemModelingFactory::createConstraintsComponent() {
//     return std::make_unique<ConstraintsConcrete>();
// }

// inline std::unique_ptr<IOptimization> SystemModelingFactory::createOptimizationComponent() {
//     return std::make_unique<OptimizationConcrete>();
// }

// inline std::unique_ptr<ILearning> SystemModelingFactory::createLearningComponent() {
//     return std::make_unique<LearningConcrete>();
// }

// inline std::unique_ptr<IPerformance> SystemModelingFactory::createPerformanceComponent() {
//     // e.g., typically a PerformanceLogger instance
//     return std::make_unique<PerformanceLogger>();
// }

// // End of ERL_Stage_1_Framework_03/Stage_01/Factories/SystemModellingFactory.h