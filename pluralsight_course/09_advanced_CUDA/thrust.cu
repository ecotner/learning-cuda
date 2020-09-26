/*
This is a demonstration of how crazily simple using `thrust` is compared
to using the lower-level runtime API.
*/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <ctime>
#include <cstdio>

int myrand() {
    return rand() % 10;
}

int main() {
    int count = 1024;
    // initialize vector on host
    thrust::host_vector<int> h(count);
    thrust::generate(std::begin(h), std::end(h), myrand);
    // copy memory from host to device; essentially the same as `cudaMemcpy`
    thrust::device_vector<int> d = h; 
    // sort data on device
    thrust::sort(std::begin(d), std::end(d));
    // copy from device back to host
    h = d;
    // print results
    for (int i=0; i<count; i++) {
        printf("%d\t", h[i]);
    }
}