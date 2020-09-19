# Tools of the trade
* CUDA Toolkit
  * LLVM-based compiler
  * Headers and libraries
  * Documentation
  * Samples
* NSight
  * IDE built on eclipse
  * not good for regular debugging
* compiling
  * use `nvcc` CLI to compile
  * can also use NSight, Visual Studio
* debugging
  * can't debug device that is providing display?
* profiling
  * can use `nvprof` on the command line
  * use `nvidia-smi` to monitor graphics card stats like power consumption, memory usage, etc