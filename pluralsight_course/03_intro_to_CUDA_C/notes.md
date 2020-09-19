# Intro to CUDA C

## Hello world
* CUDA "kernels" need to be prepended with `__global__`
* have "global" access to thread identifiers with the `threadIdx` object
    * no idea what the `.x`, `.y`, `.z` attributes mean yet
* memory management
    * need to allocate memory on device using `cudaMalloc`
    * need to copy memory to/from the GPU with `cudaMemcpy`
* actually call the kernel using weird `<<<>>>` notation
    * `func<<<blocks, threads>>>(args)` calls the kernel with `blocks` blocks and `threads` threads?
  
## Location qualifiers
* `__global__` defines a kernel that runs on the GPU, but is called from the CPU
    * executed with `<<<dim3>>>` arguments
* `__device__` runs on GPU, called from GPU
    * can be used for variables too
    * raises error when called from CPU
* `__host__` is called from CPU and runs on CPU
    * doesn't sound like it would be useful, but you can use it for testing
    * e.g. use `__host__ __device__ foo()` to compile for both CPU and GPU

## Execution model
* have lots and lots of threads
* threads are grouped together in "blocks"
* thread blocks are organized in a "grid"
    * for example, if you have a grid with two blocks of 3 threads each, you could call a function running on it like `doSomething<<<2, 3>>>()`
    * a grid with 4 blocks of 2 threads each, you could call a function running on it like `doSomething<<<4, 2>>>()`
* thread blocks are also subdivided into "warps"
    * number of warps/thread determined by compute capability
    * all warps in each block handled in parallel
* thread blocks are scheduled to run on available SM's
    * each SM executes only one block at a time
    * remember, SM's and SP's are _hardware_ resources, threads and blocks are _software_ abstractions

## Dimensions
* In `hello_world.cu`:
    * we defined execution as `<<<a, b>>>`
    * a grid of `a` blocks and `b` threads each
    * the grid and each block were 1D structures
* in reality, these constructs are 3D
    * a 3-dimensional grid of 3-dimensional blocks
    * you can define a grid of `a*b*c` blocks, each with `x*y*z` threads
    * can have 2D or 1D grids by setting extra dimensions to 1
    automatic conversion of `<<<a,b>>>` --> `(a,1,1)` by `(b,1,1)`

## Thread variables
* hold information about execution parameters and current position
* these can be accessed without initialization from any kernel
* `blockIdx`
    * where we are in the grid (e.g. `blockIdx.x` tells us the `x` coordinate of block in the grid)
* `gridDim`
    * tells us the size of the grid (e.g. `gridDim.y` tells us the size of the grid in the `y` direction)
* `threadIdx` tells us the position of the thread within the thread block
    * also has `(x,y,z)` coordinates
* `blockDim` tells us the size of the thread block
    * also has `(x,y,z)` coordinates
* limitations:
    * maximum grid/block sizes

## Error Handling
* when something fails on the GPU, it doesn't throw an error, it just FAILS SILENTLY
* core cuntions return `cudaError_t`
    * can check against `cudsSuccess`
    * get description with `cudaGetErrorString()`
* libraries may have their own error types
    * e.g. `cuRAND` has `curandStatus_t`