## History of GPU computation
* started off as dedicated hardware for computing graphics
* people realized the high degree of parallelism could be used for other purposes
* initially, you had to re-cast your numerical computations into texture/pixels/vertices so that the GPU could interpret them as graphics computations
* vendors realized that this was a new source of revenue and developed GPU's which were capable of general-purpose programming

## GPU frameworks
* CUDA (Compute Unified Driver Architecture)
  * Developed by NVIDIA (proprietary product)
  * Extensions to C/C++ programming languages
  * Wrappers for other languages (FORTRAN, PyCUDA, MATLAB)
* OpenCL (Open Computing Language)
  * Open-source project
  * Broader scope than CUDA - can be executed on GPU's, CPU's, FPGA's, etc
  * Can use to execute programs on non-NVIDA GPU's (ATI)
* C++ AMP (Accelerated Massive Programming)
  * C++ superset
  * Standardized by Microsoft, part of MSVC++
  * Supports both ATI and NVIDIA

## GPU architecture
Disclaimer: most terminology will be NVIDIA-specific
* Streaming Multiprocessor (SM)
  * contains several CUDA cores
  * can have >1 SM on card
* CUDA Core (aka Streaming Processor, Shader Unit)
  * \# of cores tied to compute capability
* Different types of memory
  * device memory
    * shared between SM's
    * read/write
  * shared memory
    * shared between SP's
    * read/write
  * constant memory
  * texture memory

## Compute capability
* a number indicating what the card can do
  * ex: 1.0, 1.x, 2.x, 3.0, 3.5 (think the most recent is like 10.x or 11.x as of this writing)
* affects hardware and API support
  * number of CUDA cores/SM
  * max \# of 32-bit registers/SM
  * max \# of instructions per kernel
  * support for double-precision ops
  * etc...
* higher is typically better
* see [Wikipedia](http://en.wikipedia.org/wiki/CUDA) for details

## Choosing a graphics card
* for numerical computations, primary spec is the peak FLOPS (FLoating-point OPerations per Second)
  * pay attention to single vs. double precision
  * if you want to do integer-valued calculations, ATI cards might be better
* consider number of cores, compute capability, memory size
* ensure PSU is good enough to handle it
* it is possible to get >1 graphics card, but there are a whole bunch of caveats
  * PCI saturation (do you have enough I/O bandwidth?)
  * power draw
* possible to mix architectures (NVIDIA + ATI)