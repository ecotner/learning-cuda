/*
this is the same file as the atomic sum computation in the previous chapter,
only now we will be profiling it using CUDA events.
*/
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

    // initialize CUDA event
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // start recording the event, run the kernel, then record another event
    cudaEventRecord(start);
    sum<<<1,count>>>(d);
    cudaEventRecord(end);
    // call this to make sure the CPU/GPU are in sync
    cudaEventSynchronize(end);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end);

    // read result back into host memory and print
    int hSum;
    cudaMemcpyFromSymbol(&hSum, dSum, sizeof(int));
    printf("The sum of numbers from 1 to %d is %d\n", count, hSum);
    printf("And it took %f msec\n", elapsed);
    cudaFree(d);
}