
void SystemCaptureFactory::initializeSystem() {
    // Create cameras, algorithms, SoC from the modelingFactory
    auto cam1 = modelingFactory_.createDataComponent();
    //auto cam2 = modelingFactory_.createDataComponent();
    auto alg1 = modelingFactory_.createAlgorithmComponent();
    auto alg2 = modelingFactory_.createAlgorithmComponent();
    auto socPtr = modelingFactory_.createSoCComponent();

    // (Optional) Set each component’s error callback to a global function
    if (errorCallback_) {
        cam1->setErrorCallback(errorCallback_);
      //  cam2->setErrorCallback(errorCallback_);
        alg1->setErrorCallback(errorCallback_);
        alg2->setErrorCallback(errorCallback_);
        socPtr->setErrorCallback(errorCallback_);
    }

    // Attempt to open camera devices
    bool cam1Opened = cam1->openDevice("/dev/video0");
    if (!cam1Opened) {
        if (errorCallback_) {
            errorCallback_("[SystemCaptureFactory] Failed to open /dev/video0");
        }
    }
/*
    bool cam2Opened = cam2->openDevice("/dev/video1");
    if (!cam2Opened) {
        if (errorCallback_) {
            errorCallback_("[SystemCaptureFactory] Failed to open /dev/video1");
        }
    }
*/
    // (Optional) Configure the cameras before streaming if needed:
    // For example:
    ////*
    CameraConfig defaultConfig;
    defaultConfig.width = 640;
    defaultConfig.height = 480;
    defaultConfig.fps = 30;
    defaultConfig.pixelFormat = "YUYV";
    if (cam1Opened) cam1->configure(defaultConfig);
  //  if (cam2Opened) cam2->configure(defaultConfig);
    //*/

    // Store all in vectors
    cameras_.push_back(std::move(cam1));
    //cameras_.push_back(std::move(cam2));
    algorithms_.push_back(std::move(alg1));
    algorithms_.push_back(std::move(alg2));
    soc_ = std::move(socPtr);

    // Initialize SoC
    soc_->initializeSoC();

    // Attempt to start streaming for each camera
    for (auto& cam : cameras_) {
        // If openDevice() worked, try startStreaming()
        if (!cam->startStreaming()) {
            if (errorCallback_) {
                errorCallback_("[SystemCaptureFactory] Failed to start streaming one camera.");
            }
        }
    }

    // Now, only spawn capture threads for cameras actually streaming
    for (auto& cam : cameras_) {
        if (cam->isStreaming()) {
            cameraThreads_.emplace_back(&SystemCaptureFactory::captureLoop, this, cam.get());
        } else {
            if (errorCallback_) {
                errorCallback_("[SystemCaptureFactory] Not spawning capture thread; streaming is off.");
            }
        }
    }

    // Start algorithms (if each AlgorithmConcrete internally spawns a thread)
    for (auto& alg : algorithms_) {
        alg->startAlgorithm();
    }

    running_ = true;

    // Spawn SoC monitoring thread
    socThread_ = std::thread(&SystemCaptureFactory::monitorLoop, this, soc_.get());

    // (Optional) Spawn orchestrator-based algorithm threads (if AlgorithmConcrete does not spawn itself)
    
    for (auto& alg : algorithms_) {
        algorithmThreads_.emplace_back(&SystemCaptureFactory::algorithmLoop, this, alg.get());
    }
    

    std::cout << "[SystemCaptureFactory] System initialized with 2 cameras, 2 algs, 1 SoC.\n";
}
