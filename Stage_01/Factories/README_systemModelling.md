Key Concepts and Optimizations

SharedQueue for Frame Distribution
• The DataConcrete component now makes a deep copy of each captured frame (wrapped in a FrameData object) and pushes it into two SharedQueues (one for display, one for the algorithm).
• Because the queue is thread-safe and bounded, the producer (camera) won’t overload consumers and will block if queues are full.

Separation of Concerns via Factories
• The SystemModelingFactory sets up core components (SoC, Data, Algorithm, and Display) and injects the same SharedQueue instances so that all modules work on the same frames.
• The SystemCaptureFactory then starts the capture loop, the algorithm loop, and (if enabled) the SoC monitoring loop. In the capture loop, the DataConcrete dequeues frames from the camera and pushes them into the display and algorithm queues.

Display Thread Doesn’t Block
• The SdlDisplayConcrete component is built to continuously poll both shared queues. It pops a frame if available; otherwise, it uses the last frame. This avoids a situation where the display thread is blocked indefinitely because no frame is coming in.
• In addition, the display loop polls SDL events so that the window remains responsive.

ThreadManager Centralizes Thread Creation and Joining
• All component threads (camera, algorithm, SoC monitor, display) are created via our ThreadManager, which ensures proper startup and shutdown.

Debugging the “No Output” Issue
• If your display never shows output, verify that (a) frames are being pushed into the display queue, and (b) the display thread is not blocked waiting for a frame.
• Adding debug logs (e.g., printing the first pixel’s RGB value after conversion) can help ensure that frames are correctly arriving in the display loop.


** Framework goal:

our SharedQueue–based pipeline into the factory framework to solve the display issue. In our framework the DataConcrete (camera capture) pushes each captured frame into two queues (one for display, one for algorithm processing). Then the SdlDisplayConcrete continuously pops frames from the display queue, converts them to RGB, and renders them.

** System Modeling Factory:

Framework Architecture Analysis
Our framework follows a Factory Pattern and is divided into five main stages, ensuring modularity and scalability. Each stage contributes to a distinct aspect of system operation.

🔹 Stage 1: SystemModelingFactory
Role: Handles the initialization and configuration of SoC components, algorithms, and data structures.
Key Components:
SoCConcrete.h: Represents the SoC hardware interface.
AlgorithmConcrete.h: Implements algorithm execution (e.g., Optical Flow).
DataConcrete.h: Manages camera input.
SystemModellingFactory.h: Factory class for setting up the system mode