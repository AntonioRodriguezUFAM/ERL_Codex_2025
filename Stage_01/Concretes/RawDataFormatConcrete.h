#pragma once
#include "../Interfaces/IRawDataFormat.h"
#include <iostream>

class RawDataFormatConcrete : public IRawDataFormat {
public:
    void initializeComponents() override {
        std::cout << "\n  [RawDataFormatConcrete] Initializing components...\n";
    }
    void formatData() override {
        std::cout << "\n  [RawDataFormatConcrete] Formatting data...\n";
    }
};
