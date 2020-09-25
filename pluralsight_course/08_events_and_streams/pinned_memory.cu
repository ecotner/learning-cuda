#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <cstdio>

/*
Repeatedly copies an array from the host to the device, but does so
using different methods depending on parameters passed.

\param pinned (bool): if true, it allocates pinned memory that is able
to be copied quickly to device. if false, normal paged memory is allocated.
\param toDevice (bool): if true, memory is copied from the host to the
device, if false, memory is copied the opposite direction.

\returns float: the number of milliseconds elapsed
*/
float timeMemory(bool pinned, bool toDevice) {
    // initialize the number of elements to copy (`count`) and the number
    // of times to do the copying (`iterations`)
    const int count = 1 << 20; // 2^20 ~ 10^6
    const int iterations = 1 << 6; // 2^6 = 64
    const int size = count * sizeof(int);

    // initialize event parameter
    cudaEvent_t start, end;
    // pointers for host/device arrays to be copied
    int *h, *d;
    // total elapsed time of the operation
    float elapsed;
    // status of CUDA errors
    cudaError_t status;

    // create events on device
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // allocate room for array on device
    cudaMalloc(&d, size);
    // allocate room for array on host
    if (pinned) {
        // use `cudaHostAlloc` to allocate pinned memory on host
        cudaHostAlloc(&h, size, cudaHostAllocDefault);
    }
    else {
        // use `malloc` or `new` to allocate regular paged memory on host
        h = (int*) malloc(size);
        // h = new int[count]; // equivalent to above line
    }
    // check to make sure memory was actually allocated properly
    if (h == 0) {
        printf("Memory could not be allocated\n");
        exit(0);
    }
    // start recording CUDA events
    cudaEventRecord(start);
    // repeatedly copy between host/device
    for (int i=0; i<iterations; i++) {
        if (toDevice) {
            status = cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
        } else {
            status = cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
        }
    }
    // stop timing, get elapsed time of copy
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);

    // free memory on host
    if (pinned) {
        // use `cudaFreeHost` if memory was pinned using `cudaHostAlloc`
        cudaFreeHost(h);
    } else {
        // otherwise just delete as normal
        free(h);
        // delete [] h; // equivalent
    }
    // free memory on device
    cudaFree(d);
    // free events? what does this do?
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    
    // return the total time elapsed running memory copy
    return elapsed;
}

int main() {
    // run memory copy profiling for each combination of parameters
    printf("From device, paged memory:\t%f ms\n", timeMemory(false, false));
    printf("From device, pinned memory:\t%f ms\n", timeMemory(true, false));
    printf("To device, paged memory:\t%f ms\n", timeMemory(false, true));
    printf("To device, pinnned memory:\t%f ms\n", timeMemory(true, true));
}

/*
These are the results I get running on my device:

From device, paged memory:      36.702175 ms
From device, pinned memory:     22.142656 ms
To device, paged memory:        33.218559 ms
To device, pinnned memory:      22.517759 ms
*/