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

    // initialize stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // declare host/device arrays
    float *ha, *hb, *hc, *da, *db, *dc;

    // allocate memory
    const int totalSize = totalCount * sizeof(float);
    const int chunkSize = chunkCount * sizeof(float);
    cudaMalloc(&da, chunkSize);
    cudaMalloc(&db, chunkSize);
    cudaMalloc(&dc, chunkSize);
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
    cudaEventRecord(start, stream);
    // split data into chunks and iterate over them
    for (int i=0; i<totalCount; i+=chunkCount) {
        // copy pinned memory from host to device without blocking
        cudaMemcpyAsync(da, ha+i, chunkSize, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(db, hb+i, chunkSize, cudaMemcpyHostToDevice, stream);
        // execute the kernel
        kernel<<<chunkCount/64,64,0,stream>>>(da, db, dc);
        // copy result back to host
        cudaMemcpyAsync(hc+i, dc, chunkSize, cudaMemcpyDeviceToHost, stream);
    }
    // wait until stream reaches here, record end event
    cudaStreamSynchronize(stream);
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
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaStreamDestroy(stream);
}

// executing this on my device results in:
// This took 8.373248 ms