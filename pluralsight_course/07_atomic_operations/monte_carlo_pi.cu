#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <cstdio>
#include <ctime>

// global counter to count points that fall into circle
__device__ int dnum = 0;

__global__ void countPoints(float* xs, float* ys) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float x = xs[idx];
    float y = ys[idx];
    int n = (x*x + y*y < 1.0f) ? 1 : 0;
    // int n = 1;
    atomicAdd(&dnum, n);
}

int main() {
    // number of points that we're going to generate
    const int count = 512*512; // 262144
    const int size = count * sizeof(float);
    // status/error variables?
    cudaError_t cudaStatus;
    curandStatus_t curandStatus;
    // random number generator
    curandGenerator_t gen;

    // initialize random number generator
    curandStatus = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, time(0));

    // allocate memory on device for (x,y) coordinates of points
    float *x, *y;
    cudaStatus = cudaMalloc(&x, size);
    cudaStatus = cudaMalloc(&y, size);

    // generate a bunch of random numbers for x and y between [0, 1]
    curandStatus = curandGenerateUniform(gen, x, count);
    curandStatus = curandGenerateUniform(gen, y, count);

    // count the points that fall inside the circle
    countPoints<<<512,512>>>(x, y);

    // copy the result back to host
    int hnum;
    // why does function declaration say `dnum` should be a pointer, but
    // it doesn't work when I pass `&dnum`? does `__device__` implicitly
    // declare a pointer?
    cudaMemcpyFromSymbol(&hnum, dnum, sizeof(int));
    cudaFree(x);
    cudaFree(y);

    // print result
    float pi = 4.0f * ((float)hnum / (float)count);
    printf("pi is approximately %f\n", pi);
    return cudaStatus | curandStatus;
    // compile with `nvcc -lcurand monte_carlo_pi.cu` to include curand lib
}