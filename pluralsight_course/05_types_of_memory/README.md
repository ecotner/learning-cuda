# Types of memory
This refers to RAM on the _device_, not on the host, or any other types of memory like HDD/SSD, etc.

## Global memory
* has "grid scope" (available to all threads in all blocks in the grid)
* application lifetime (exists until the app exits or explicitly deallocated)
* dynamic
    * `cudaMalloc` to allocate
    * pass pointer to kernel
    * `cudaMemcpy` to copy to/from host memory
    * `cudaFree` to deallocate
* static
    * declare global variable as device (e.g. `__device__ int sum = 0;`)
    * use freely within the kernel
    * use `cudaMemcpy[To/From]Symbol()` to copy to/from host memory
    * no need to explicitly deallocate
* slowest and most inefficient, but probably the most flexible

## Constant & texture memory
* memory that will not change over the course of execution
    * also, CANNOT be dynamically allocated
* useful for lookup tables, model parameters, etc.
* grid scope, application lifetime
* resides in device memory, but...
* cached in a constant memory cache
* constrained by `MAX_CONSTANT_MEMORY`
    * expect \~ 64 kB typically
* similar operation to statically-defined device memory
    * declare as `__constant__`
    * use freely within the kernel
    * use `cudaMemcpy[To/From]Symbol()` to copy to/from host memory
* very fast provided all the threads read from the same location
    * had to google this; what it means is that each time a warp pulls from the memory, the cost is linear in the number of distinct addresses. so if all the threads in a warp pull from different addresses, it is 32 times _slower_ than if they were pulling from a single address and sending that info to all the threads in the warp
* used for kernel arguments `<<<a,b>>>`
* texture memory: similar to constant memory, optimized for 2D access

## Shared memory
* block scope
    * shared only within a thread block
    * not shared between blocks
    * limited to 49152 bytes/block (on my GTX 1080)
* kernel lifetime
    * must be declared within the kernel function body, and goes away after the kernel has exited
* very fast compared to global memory
* pretty comprehensive [blog post](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) on NVIDIA site

## Register and local memory
* memory that is being allocated while the kernel is being executed
* has thread scope, kernel lifetime
* non-array memory
    * like the `int tid` variables we've been using
    * stored in a register
    * very fast
* array memory
    * stored in "local memory"
    * local memory is an abstraction, actually put in global memory
    * thus, it is just as slow as global memory

## Summary

Declaration | Memory | Scope | Lifetime | Slowdown
--- | --- | --- | --- | ---
`int foo;` | register | thread | kernel | 1x
`int foo[10];` | "local" | thread | kernel | 100x
`__shared__ int foo;` | shared | block | kernal | 1x
`__device__ int foo;` | global | grid | application | 100x
`__constant__ int foo;` | constant | grid | application | 1x