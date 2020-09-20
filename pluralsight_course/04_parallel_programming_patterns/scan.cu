#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

// kernel to compute cumulative sum
__global__ void runningSum(int * d) {
    int threads = blockDim.x;
    int tid = threadIdx.x;

    // tc = thread count allowed to participate
    // drop # of threads according to step size, then double step each iteration
    for (int tc=threads, step=1; tc>0; tc-=step, step*=2) {
        // only execute if in the allowed thread pool
        if (tid < tc) {
            d[tid + step] += d[tid]; 
        }
    }
}

int main() {
    // initialize elements to sum
    const int count = 16;
    const int size = count * sizeof(int);
    int h[count];
    for (int i=0; i<count; i++) {
        h[i] = i+1;
    }

    // initialize array on device
    int *d;
    cudaMalloc(&d, size);
    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    // run the calculation
    runningSum<<<1,count-1>>>(d);

    // copy results back to host
    cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
    cudaFree(d);
    d = 0;

    // print results
    for (int i=0; i<count; i++) {
        printf("%d ", h[i]);
    }
}