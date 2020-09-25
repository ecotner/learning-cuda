#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cmath>
#include <ctime>

const int chunkCount = 1 << 20; // 2^20 ~ 10^6
const int totalCount = chunkCount << 3; // 2^23 ~ 8*10^6

// add two numbers together and take error function of result, store in array
__global__ void kernel(float* a, float* b, float* c) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < chunkCount)
        c[tid] = erff(a[tid] + b[tid]);
}

int main() {
    // get device properties
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    // if device overlap is not possible, we can't do this demo
    if (!prop.deviceOverlap) {
        printf("Device does not have GPU_OVERLAP\n");
        exit(0);
    }


    // initialize events
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // initialize streams
    // *** note that we have TWO streams now ***
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);

    // declare host/device arrays
    // since we have two streams (and don't want thread collisions),
    // we'll need two copies of each device array
    float *ha, *hb, *hc, *d1a, *d1b, *d1c, *d2a, *d2b, *d2c;

    // allocate memory
    const int totalSize = totalCount * sizeof(float);
    const int chunkSize = chunkCount * sizeof(float);
    cudaMalloc(&d1a, chunkSize);
    cudaMalloc(&d1b, chunkSize);
    cudaMalloc(&d1c, chunkSize);
    cudaMalloc(&d2a, chunkSize);
    cudaMalloc(&d2b, chunkSize);
    cudaMalloc(&d2c, chunkSize);
    // use pinned memory here for faster data transfer.
    // we will be doing multiple transfers because of the
    // chunking, so it will be worth the allocation overhead.
    cudaHostAlloc(&ha, totalSize, cudaHostAllocDefault);
    cudaHostAlloc(&hb, totalSize, cudaHostAllocDefault);
    cudaHostAlloc(&hc, totalSize, cudaHostAllocDefault);

    // fill a and b with some random values
    srand((unsigned) time(0));
    for (int i=0; i < totalCount; i++) {
        // generate random numbers between [0,1]
        ha[i] = rand()/RAND_MAX;
        hb[i] = rand()/RAND_MAX;
    }

    // start recording event stream
    cudaEventRecord(start, stream1);
    // split data into chunks and iterate over two chunks at a time (interleaving the two streams)
    for (int i=0; i<totalCount; i+=2*chunkCount) {
        int i1 = i;
        int i2 = i + chunkCount;
        cudaMemcpyAsync(d1a, ha+i1, chunkSize, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d2a, ha+i2, chunkSize, cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d1b, ha+i1, chunkSize, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d2b, ha+i2, chunkSize, cudaMemcpyHostToDevice, stream2);
        kernel<<<chunkCount/64,64,0,stream1>>>(d1a, d1b, d1c);
        kernel<<<chunkCount/64,64,0,stream2>>>(d2a, d2b, d2c);
        cudaMemcpyAsync(hc+i1, d1c, chunkSize, cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(hc+i2, d2c, chunkSize, cudaMemcpyDeviceToHost);
    }
    // wait until streams reach here, record end event
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    // get total elapsed time
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end);

    // print results
    printf("This took %f ms\n", elapsed);

    // free memory
    cudaFreeHost(ha);
    cudaFreeHost(hb);
    cudaFreeHost(hc);
    cudaFree(d1a);
    cudaFree(d1a);
    cudaFree(d1b);
    cudaFree(d2c);
    cudaFree(d2b);
    cudaFree(d2c);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

// executing this on my device results in:
// This took 8.091616 ms

// compare this with the single-stream application, which had an
// elapsed time of 8.373248 ms (3.4% reduction). it isn't a super
// crazy speedup, but it is a speedup nonetheless thanks to streams