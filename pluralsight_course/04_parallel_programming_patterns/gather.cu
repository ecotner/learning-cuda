/*
This file shows how to use the "gather" pattern in CUDA. This specific example is using the
Black-Scholes equation, where we parallelize over securities, but gather parameters for each
security.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h" // CUDA random number generators
#include <math.h> // standard mathematical operations (log, erf, sqrt)
#define _USE_MATH_DEFINES // common mathematical constants (`M_SQRT1_2`, used below, is equal to sqrt(1/2))
#include <cstdio>

// cumulative distribution function (CDF) of a standard normal distribution
// note the `__inline__`, which tells the compiler to just inline the function at compile time for
// performance (at the cost of larger binary size)
__device__ __host__ __inline__ float N(float x) {
    return 0.5 + 0.5 * erff(x * M_SQRT1_2);
}

// options are a right to buy (call) or sell (put) an asset at a specific price/date
// k = strike price, s = underlying asset price, t = time until option expires,
// r = rate at which money can be borrowed, v = volatility of option
// c = call price, p = put price
// this kernel actually does all the calculations for each security
__device__ __host__ void price(float k, float s, float t, float r, float v, float* c, float* p) {
    float srt = v * sqrtf(t);
    float d1 = (logf(s/k) + (r+0.5*v*v)*t) / srt;
    float d2 = d1 - srt;
    float kert = k * expf(-r*t);
    *c = erff(d1)*s - erff(d2)*kert;
    *p = kert - s + *c;
}

// intermediate kernel which selects the index of the appropriate security and passes on
// computation to the function defined above
__global__ void price(float* k, float* s, float* t, float* r, float* v, float* c, float* p) {
    int idx = threadIdx.x;
    price(k[idx], s[idx], t[idx], r[idx], v[idx], &c[idx], &p[idx]);
}

int main() {
    const int count = 512; // number of securities to analyze
    const int size = count * sizeof(float);
    float *args[5]; // array of arrays of parameters for each security (k, s, t, r, v)

    // random generator for security parameters
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    // generate values for the parameters directly on device
    for (int i=0; i<5; i++) {
        cudaMalloc(&args[i], size);
        curandGenerateUniform(gen, args[i], count);
    }

    float *dc, *dp; // call and put arrays
    cudaMalloc(&dc, size); // allocate space on device
    cudaMalloc(&dp, size);
    // just realized - `&dc` and `&dp` are pointers to pointers... passing them to `cudaMalloc` allows
    // the function to overwrite the pointers `dc` and `dp` to point at _new_ memory locations on the
    // device, rather than the original locations on the host they were pointed at when declared

    // calculate call/put values
    price<<<1,count>>>(args[0], args[1], args[2], args[3], args[4], dc, dp);

    // copy memory from device to host
    float hc[count], hp[count];
    cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hp, dp, size, cudaMemcpyDeviceToHost);

    // free memory on device
    cudaFree(&dc);
    cudaFree(&dp);
    dc = NULL; dp = NULL; // make sure pointers can't be used or freed a second time
    for (int i=0; i<5; i++) {
        cudaFree(&args[i]);
        args[i] = NULL;
    }

    // print out some values
    for (int s=0; s<10; s++) {
        for (int i=0; i<5; i++) {
            printf("Call price: $%.2f, put price: $%.2f\n", hc[i], hp[i]);
        }
    }

    return 0;
}