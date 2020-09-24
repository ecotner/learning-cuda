#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// header file where `atomicAdd` is defined (or maybe not?)
#include "sm_60_atomic_functions.h"
#include <cstdio>

// declare some global memory on the device
__device__ int dSum = 0;

__global__ void sum(int* d) {
    int tid = threadIdx.x;
    // this would be a naiive way to increment the value, but results in threads writing
    // multiple different values to the same memory location, causing races
    // dSum += d[tid]; 
    // this blocks all other threads so that only one thread at a time may modify the `dSum` variable
    atomicAdd(&dSum, d[tid]);
}

int main() {
    // initialize a vector of integers
    const int count = 128;
    const int size = count * sizeof(int);
    int h[count];
    for (int i=0; i<count; i++) {
        h[i] = i+1;
    }


    // copy that vector over to the device
    int* d;
    cudaMalloc(&d, size);
    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    // run the kernel
    sum<<<1,count>>>(d);

    // read result back into host memory and print
    int hSum;
    cudaMemcpyFromSymbol(&hSum, dSum, sizeof(int));
    printf("The sum of numbers from 1 to %d is %d\n", count, hSum);
    cudaFree(d);
}