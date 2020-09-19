#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// adds array elements in a loop like normal
void addArrays(int* a, int* b, int* c, int count) {
    for (int i = 0; i < count; i++) {
        c[i] = a[i] + b[i];
    }
}

// simulates adding each element in a separate thread indexed by `i`
void addArraysThread(int* a, int* b, int* c, int i) {
    c[i] = a[i] + b[i];
}


// CUDA-compatible "kernel" (aka function) prepended with `__global__`
// notice thread number/ID not needed to be passed in as argument
__global__ void addArraysCUDA(int* a, int* b, int* c) {
    int i = threadIdx.x; // thread index available "globally"
    c[i] = a[i] + b[i];
}

int main() {
    const int count = 5;
    const int size = count * sizeof(int);
    // the `h` prepending each variable shows that it is memory that sits on the "host" (CPU)
    int ha[] = {1,2,3,4,5};
    int hb[] = {10,20,30,40,50};
    int hc[count];
    // memory that sits on the "device" (GPU)
    int *da, *db, *dc;
    // allocate memory (on device?) using `cudaMalloc`
    cudaMalloc(&da, size); // if `da` is already a pointer, why use `&da` to get address?
    cudaMalloc(&db, size);
    cudaMalloc(&dc, size);

    // copy memory from host to device
    // cudaMemcpy(*dest, *src, size_t size, kind) // the "kind" says whether you're copying from host -> device or vice versa
    cudaMemcpy(da, ha, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyKind::cudaMemcpyHostToDevice);

    // addArrays(a, b, c, count); // this does the computation all at once

    // for (int i=0; i < count; i++) { // simulates different threads (not actually parallel)
    //     addArraysThread(a, b, c, i);
    // }

    // this is how you call CUDA kernels??
    addArraysCUDA<<<1, count>>>(da, db, dc); // 1 block with `count` threads

    // copy memory from device back to host
    cudaMemcpy(hc, dc, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    // print results
    for (int i=0; i < count; i++) {
        printf("%d ", hc[i]);
    }
    printf("\n");
}