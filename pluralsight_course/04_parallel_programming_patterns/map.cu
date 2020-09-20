#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h" // CUDA PRNG library!

#include <ctime>
#include <cstdio>

__global__ void addTen(float* d, int count) {
    int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z; // 512
    // int blocksPerGrid = gridDim.x * gridDim.y * gridDim.z; // 256; don't think this is actually needed
    // get thread position within thread block
    // x is our highest granularity index; threadIdx.x += 1 --> threadPosInBlock += 1
    // y is a lower granularity index; threadIdx.y += 1 --> threadPosInBlock += blockDim.y
    // z is our lowest granularity index; threadIdx.z += 1 --> threadPosInBlock += blockDim.y * blockDim.z
    int threadPosInBlock = threadIdx.x + blockDim.y * (threadIdx.y + blockDim.z * threadIdx.z);
    // have to do same thing for block position in grid
    int blockPosInGrid = blockIdx.x + gridDim.y * (blockIdx.y + gridDim.z * blockIdx.z);
    // index into elements of array
    int idx = threadPosInBlock + threadsPerBlock * blockPosInGrid;
    // actually do computation
    d[idx] += 10.0f;
}

int main() {
    // create random number generator and set seed
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32); // "MTGP32" = 32-bit Merseinne twister
    curandSetPseudoRandomGeneratorSeed(gen, time(0));

    const int count = 123456; // number of values to be generated
    const int size = count * sizeof(int);
    float *d; // pointer to array on device (memory not yet allocated)
    float h[count]; // empty array on host (with memory allocated)
    cudaMalloc(&d, size); // allocate memory to array on device
    curandGenerateUniform(gen, d, count); // generate random uniform values, fill in array `d` on device

    // we want to perform each kernel operation on the entire array at once
    dim3 block(8, 8, 8); // define 8*8*8 thread block with 512 threads
    // (123456 threads) / (512 threads/block) = 241.1 blocks
    dim3 grid(16, 16); // define grid with 16*16 = 256 blocks, just enough to handle all the blocks
    // do the calculation
    addTen<<<grid, block>>>(d, count);
    // copy memory back to host
    cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
    cudaFree(&d);
    // print some results
    for (int i=0; i<10; i++) {
        printf("%f\n", h[i]);
    }
}