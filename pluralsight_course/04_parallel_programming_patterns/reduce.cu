#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <math.h>

// this is a function the instructor came up with
__global__ void sumSingleBlock(int* d) {
    int tid = threadIdx.x;
    // iterate over reduce steps
    // recall `>>=` does a left bitshift/assignment, so this is a clever
    // way of halving the thread count `tc` on each step
    for (int tc=blockDim.x, stepSize=1; tc>0; tc>>=1, stepSize<<=1) {
        if (tid < tc) {
            int pa = tid * stepSize * 2;
            int pb = pa + stepSize;
            d[pa] += d[pb];
        }
    }
}

__global__ void mySumSingleBlock(int* d) {
    int tid = threadIdx.x;
    // iterate over aggregation steps; keep track of thread count `tc`,
    // the number of threads still doing useful operations
    for (int tc=blockDim.x, step=1; tc > 0; tc/=2, step*=2) {
        if (tid < tc) { // only have participating threads do useful work
            // map thread to array positions it will sum together
            int pa = tid * step * 2;
            int pb = pa + step;
            d[pa] += d[pb];
        }
    }
}

int main() {
    // can't go any higher than 2^10 = 1024 threads/block on my GPU (GTX 1080),
    // where the max blockDim.x == 1024.
    // however, when using int32, the sum from 1..1024 is 1024*1023/2 = 523776,
    // which is larger than 2^15-1 = 32767, the maximum int size, so the
    // computation will overflow anyway. hence we use 512, which has a sum from
    // 1..512 of 512*511/2 = 131328... which is still higher than the max...
    // how is this giving the correct output? is this compiler-specific?
    const int count = 512;  
    printf("%d elements\n", count);
    const int size = count * sizeof(int);

    int h[count];
    for (int i=0; i<count; i++) {
        h[i] = i+1;
    }

    int *d;
    cudaMalloc(&d, size);
    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    // sumSingleBlock<<<1,count/2>>>(d);
    mySumSingleBlock<<<1,count/2>>>(d);

    cudaMemcpy(h, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);

    printf("Sum is %d\n", h[0]);
}