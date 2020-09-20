# Parallel programming patterns

## Rules of the game
* Different types of memory
    * global vs. local
    * access speeds
* Data is in arrays
    * No parallel data structures (is this still true? libraries like [Thrust](https://developer.nvidia.com/thrust) exist now)
    * No collections (vector, set, etc.) (check out Thrust ^^^)
    * No auto-parallelization/vectorization compiler support
    * No CPU-type SIMD equivalent
* Compiler constraint
    * instructor says only C++03 is supported
    * pretty sure this is [out-of-date](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-cplusplus-language-support)

## Element addressing
* thread space can be up to 6D
    * 3D grid of 3D thread blocks
* input space typically 1D or 2D
* need to map inputs to threads
* some examples:
    * 1 block, N threads --> `threadIdx.x`
    * 1 block, MxN threads --> `threadIdx.y * blockDim.x + threadIdx.x`
    * N blocks, M threads --> `blockIdx.x * gridDim.x + threadIdx.x`
    * ...

## Map
* Applying a function to an array, and replicating that function over every element in that array
* $y_i = f(x_i)$
* have to map threads from 6D (thread, block) space into smaller # of dimensions

## Gather
* apply a function to an arbitrary selection of input values to get an output value for each group
* $y_i = f(a_m, b_n, c_p, ...)$

Will learn about this in the context of the Black-Scholes equation
* differential equation for calculating the price of an asset
* under certain conditions (no external forcing), has an analytic solution:
$$
C = N(d_1)S - N(d_2) K e^{-rt} \\
P = K e^{-rt} - S + C \\
d_1 = \left(\ln(S/K) + (r + \sigma^2/2)t\right)/\sigma\sqrt{t} \\
d_2 = d_1 - \sigma \sqrt{t} \\
N(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x dt\, e^{-t^2/2} = \frac{1}{2} \left[1 + \mathrm{erf}(x/\sqrt{2})\right]
$$

## Scatter
* Essentially the opposite of gather
* Produces multiple outputs from a single input
* $(a, b, c) = f(x)$
* have to be aware of thread collisions if output writes to multiple places in memory
    * e.g. if $(y_i, y_{i+1}, ..., y_{i+n}) = f(x_i)$, then calling $f(x_{i+1})$ can overwrite the previous results
    * or $f(x_i)$ and $f(x_{i+1})$ could try to write to the same memory location at the same time, causing problems!
    * will have to use thread synchronization techniques to fix

## Reduce
* Takes the output of a large number of threads and combines them together to produce single output
* $Y = f(x_1, x_2, ..., x_N)$
* e.g. sum of $N$ numbers $x_i$: $X = \sum_{i=1}^{N} x_i$
    * can be expressed as $(((x_1 + x_2) + x_3) + ...)$ where each $()$ is the output of a single thread
        * requires each successive thread to wait on the output of previous one... not parallelizable
    * because $+$ is associative, can break sum up into multiple groups $g \in G$:
        * $X = \sum_{g=1}^{|G|} \left[ \sum_{i=(g-1)N/|G|+1}^{gN/|G|} x_i \right]$
        * everything in the inner sum can be done on a single thread
        * once all inner sums are complete, the outer sum can be computed as a final step
    * no reason to stop there; why not create a tree hierarchy where each thread sums a single pair of $x_i$, then the result is passed onto another thread that combines that output with the output of another thread, that is then passed onto another thread...
    * $y_{i,j+1} = y_{2i-1,j} + y_{2i,j}$ --> start where $y_{i,0} = x_i$ (the actual data you're summing), then plug in recursively (where $i \in [1, |X|/2^j]$)

## Scan
* Each output value is a sum of previous values: $y_n = \sum_{i=1}^n x_i$
* Sounds kind of sequential, so how can we parallelize it?
    * Similar to the reduce pattern, except computation is a DAG instead of a tree; previous results get reused more often
    * Steps:
        1. Start off with N-1 threads, and step size $s=1$
        2. use thread $i$ to add elements $i$ and $i+s$ together, overwrite element $i+s$
        3. discard $s$ threads (from the top of the sequence), then double $s$
        4. repeat 2-3 until finished