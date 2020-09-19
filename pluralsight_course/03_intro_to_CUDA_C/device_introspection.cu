/*
We can look up GPU parameters (max grid/block size, # of warps, etc) on NVIDIA's website:
    https://developer.nvidia.com/cuda-toolkit-archive
But what if we want to access these properties at runtime (so we don't hardcode our program
to only be compatible with a single GPU)? We can use a variety of useful functions and
objects that can get these for us. Documentation can be found at
    https://docs.nvidia.com/cuda/
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
using std::cout;

int main() {
    int count;
    cudaGetDeviceCount(&count); // puts number of devices into `count`

    cudaDeviceProp prop; // declares a property object
    for (int i=0; i<count; i++) { // iterates through devices
        cudaGetDeviceProperties(&prop, i); // populates properies for each device
        cout << "Device " << i << ": " << prop.name << '\n';
        cout << "Compute capability: " << prop.major << "." << prop.minor << '\n';
        cout << "Max grid dimensions: (" <<
            prop.maxGridSize[0] << " x " <<
            prop.maxGridSize[1] << " x " <<
            prop.maxGridSize[2] << ")\n";
        cout << "Max block dimensions: (" <<
            prop.maxThreadsDim[0] << " x " <<
            prop.maxThreadsDim[1] << " x " <<
            prop.maxThreadsDim[2] << ")\n";
        cout << "Multiprocessors: " << prop.multiProcessorCount << '\n';
        cout << "Clock Rate: " << prop.clockRate << " kHz\n";
    }
}