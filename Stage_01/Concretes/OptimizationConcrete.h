#pragma once
#include "../Interfaces/IOptimization.h"
#include <iostream>

class OptimizationConcrete : public IOptimization {
public:
    void optimizeResources() override {
        std::cout << "\n  [OptimizationConcrete] Optimizing Resources...\n";
    }
};
