#pragma once

#include <iostream>

struct Engine {
    void allocate() {
        std::cout << "I can allocate memory!!!" << std::endl;
    }
};