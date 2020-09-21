#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <math.h>

__global__ void sumSingleBlockSharedMem(int* d) {
    // declare that we're going to use shared memory in this kernel
    extern __shared__ int dcopy[];
    int tid = threadIdx.x;
    // copy the memory over from global memory to shared memory
    dcopy[tid*2] = d[tid*2];
    dcopy[tid*2+1] = d[tid*2+1];

    for (int tc=blockDim.x, stepSize=1; tc>0; tc>>=1, stepSize<<=1) {
        if (tid < tc) {
            int pa = tid * stepSize * 2;
            int pb = pa + stepSize;
            // now we need to change to use the shared memory instead of the global memory when doing the reduce
            // d[pa] += d[pb];
            dcopy[pa] += dcopy[pb];
        }
    }
    // copy the final output value over to global memory
    if (tid == 0) {
        d[0] = dcopy[0];
    }
}


int main() {
    const int count = 32;  
    printf("%d elements\n", count);
    const int size = count * sizeof(int);

    int h[count];
    for (int i=0; i<count; i++) {
        h[i] = i+1;
    }

    int *d;
    cudaMalloc(&d, size);
    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    sumSingleBlockSharedMem<<<1, count/2, size>>>(d); // third argument is the size of the shared memory

    cudaMemcpy(h, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);

    printf("Sum is %d\n", h[0]);
}