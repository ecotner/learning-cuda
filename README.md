# Learning CUDA
Learning GPU programming with CUDA

## Some resources:
* Tutorials:
    * [TutorialsPoint](https://www.tutorialspoint.com/cuda/index.htm) introduction to CUDA
    * [PluralSight](https://app.pluralsight.com/library/courses/parallel-computing-cuda/table-of-contents) introduction to CUDA
    * NVIDIA [blog post](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
    * [Presentation](https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf) on CUDA C/C++ basics
* [CUDA By Example](https://developer.nvidia.com/cuda-example)
    * apparently a good book; NOT free
* [Accelerated libraries](https://developer.nvidia.com/gpu-accelerated-libraries) for common operations
    * linear algebra, FFT, RNG, algebraic equations, sparse matrices, tensor ops, linear solvers, graph algorithms
    * check this out before rolling your own algos!
* [Thrust](https://developer.nvidia.com/thrust)
    * high-level library of parallel algorithms and data structures

## Tips:
* `vscode` intellisense w/ CUDA:
    * in `vscode`, there is an extension called `vscode-cudacpp` that gets rid of the syntax error highlighting you would get from `__global__`, `kernel<<<a,b>>>`, etc. in base C/C++
    * `vscode` does not autocomplete from the cuda libraries by default, so you need to
        1. add association of `.cu` or `.cuh` files with C++ in your `settings.json` (i.e. do [this](https://stackoverflow.com/a/62848299/8078494), but replace `"cuda"` --> `"cpp"`)
        2. in your C++ language properties `c_cpp_properties.json`, add `/usr/local/cuda/include/**` (or wherever your CUDA headers are) to your include path
* CUDA-X libraries
    * add `-lcurand` etc. to your `nvcc` commands to tell linker where to look
* device specs
    * can find simple list of device specifications on [wikipedia](https://en.wikipedia.org/wiki/GeForce_10_series#GeForce_10_(10xx)_series)
    * a more complete specification can be found on NVIDIA's documentation, but is difficult to find
    * or you can [query your device itself](https://gist.github.com/nelson-liu/623eb54d977c98db005eaf2fbc449238#gistcomment-2357284) by navigating to `~/cuda/samples/1_Utilities/deviceQuery`, build using `make`, and run `./deviceQuery`