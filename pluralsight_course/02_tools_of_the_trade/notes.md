# Tools of the trade
* CUDA Toolkit
  * LLVM-based compiler
  * Headers and libraries
  * Documentation
  * Samples
* NSight
  * IDE built on eclipse
  * not good for regular debugging
* compilation
  * use `nvcc` CLI to compile
    * not actually a compiler; splits code into GPU and non-GPU parts, then passes to native compiler
    * translates code written in CUDA C --> PTX format, then graphics driver turns PTX --> binary
  * PTX is "assembly language" of CUDA
    * intermediate bytecode generated from source code
    * possible to write "inline" PTX code in your source
* debugging
  * can't debug device that is providing display?
* profiling
  * can use `nvprof` on the command line
  * use `nvidia-smi` to monitor graphics card stats like power consumption, memory usage, etc