# Advanced CUDA

## Inline PTX
* PTX is the "assembly language" of CUDA
* you can output PTX code from your kernel
    * `nvcc -ptx`
* can load a PTX kernel with Driver API
* can embed PTX directly into kernel
    * `asm("mov.u32 %0, %%laneid;" : "=r"(laneid));`
    * splices PTX right into your kernel
    * allows referencing variables

## Driver API
* CUDA API's
    * Runtime API (what we've been using)
    * Driver API
        * `cuda.h`, `cuda.lib`
* Driver API
    * allows low-level control of CUDA
    * no "syntactic sugar (`<<<>>>`, `dim3`, etc)
    * can be mixed with runtime API
    * not useful to CUDA users in most cases

## Pinned memory
* `cudaHostAlloc(pHost, size, flags)`
* `flags` parameter can be
    * `cudaHostAllocMapped`
        * maps memory directly into GPU address space
        * lets you access host memory directly from GPU
        * aka "zero-copy memory"
        * use `cudaHostGetDevicePointer()` to get device address
    * `cudaHostAllocPortable`
        * ordinary pinned memory is visible to one host thread
        * portable pinned memory is allowed to migrate between host threads
    * `cudaHostAllocWriteCombined`
        * write-combined memory transfers faster across the PCI bus
        * cannot be read efficiently by CPU's
* can use any combination of these flags

## Multi-GPU programming
* execute parts on separate devices
    * split the work
    * execute kernels on separate threads
    * combine results
* use `cudaSetDevice(id)` to select device to run on
* portable zero-copy memory useful for multi-threading

## Thrust library
* STL-like library for accelerated computation
    * useful for automatically managing memory
* included with CUDA
* `host_vector` and `device_vector`
    * assign, resizes, etc. (each `d[n] = z;` causes a `cudaMemcpy` behind the scenes)
    * copy with `=` operator
    * can interop with STL containers and CUDA raw memory
* predefined algorithms
    * search, sort, copy, reduce
* functor syntax
    * `thrust::transform(x.begin(), x.end(), y.begin(), y.end(), thrust::multiplies<float>());`