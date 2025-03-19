// #include <cuda_runtime.h>
#include <iostream>

#include "RBAControl.h"

using namespace std;

void RBAControl::init() {
    cout << "RBAControl::init()" << endl;
    cudaSetDevice(0);
}

public:
     void shineRBG(int r, int g, int b) {
    cout << "RBAControl::shineRBG(" << r << ", " << g << ", " << b << ")" << endl;
}