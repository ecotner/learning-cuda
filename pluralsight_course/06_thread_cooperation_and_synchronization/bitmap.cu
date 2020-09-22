#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>
#include <fstream>

using std::ofstream;

#define _USE_MATH_DEFINES // has constants like `M_PI`

__global__ void kernel(unsigned char* src) {
    // create shared memory so that multiple threads can write to it
    // faster than using global memory?
    __shared__ float temp[16][16];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // create some sinusoidal density field for a pretty image
    const float period = 128.0;
    temp[threadIdx.x][threadIdx.y] = 255 *
        (sinf(x*2.0f*M_PI/period) + 1.0f) *
        (sinf(y*2.0f*M_PI/period) + 1.0f) / 4.0f;

    // write the pixel value back into the array taking into account the
    // fact that each element represents one byte of an RGBA pixel
    src[offset*4] = 0; // red
    // note that here we are deliberately causing a synchronization issue because
    // while the current thread has already written a value to `temp` in the position
    // specified by its `threadIdx`, when it tries to read the value from a different
    // location in `temp` (namely 15 minus its thread coordinates). since this location
    // is written to by a different thread, the computaton above might not have been
    // completed yet, so it will write whatever random crap `temp` was initialized to
    // to `src` instead of the intended value. we can fix this by placing a thread
    // barrier just before this line so that we ensure that all threads have completed
    // the pixel value calculation above
    __syncthreads();
    // src[offset*4+1] = temp[threadIdx.x][threadIdx.y]; // green; use this one to get really smooth, non-flipped image
    src[offset*4+1] = temp[15-threadIdx.x][15-threadIdx.y]; // green; flips the x/y coordinates within each thread block
    src[offset*4+2] = 0; // blue
    src[offset*4+3] = 255; // opacity
}

int main() {
    // size of square canvas
    int dimension = 512;
    int size = dimension * dimension * 4; // each of RGBA requires 1 byte apiece for each pixel
    // declare the place we will put the image in on host
    unsigned char dst[size];
    // allocate memory for image on device
    unsigned char* src;
    cudaMalloc(&src, size);

    // split image up into 16*16 blocks of threads
    dim3 blocks(dimension/16, dimension/16);
    dim3 threads(16, 16);
    // call kernel, making sure to pass `src` so it knows where to save the output
    kernel<<<blocks, threads>>>(src);

    // copy image back to host and release
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    cudaFree(src);
    src = 0;

    // write results to a csv file that we'll visualze later with python
    // there's probably a more efficient way to do this
    ofstream fp;
    fp.open("image.csv");
    fp << "r,g,b,a\n";
    for (int i=0; i<dimension*dimension; i++) {
        fp << (int) dst[4*i] << ',' // r
            << (int) dst[4*i+1] << ',' // g
            << (int) dst[4*i+2] << ',' // b
            << (int) dst[4*i+3] << '\n'; // a
    }
    fp.close();
}