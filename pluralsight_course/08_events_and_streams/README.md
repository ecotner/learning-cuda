# Events and streams

## Events
* how to measure performance?
* use OS timers
    * too much noise
* use profiler
    * times only kernel duration + other invocations
* CUDA events
    * event = timestamp
    * timestamp recorded on the GPU
    * invoked from the CPU side

## Event API
* `cudaEvent_t e`: the event handle
* `cudaEventCreate(&e)`: creates the event
* `cudaEventRecord(e, 0)`
    * records the event, timestamp, etc.
    * second param is the _stream_ to which to record
* `cudaEventSynchronize(e)`
    * CPU and GPU are async, can be doing things in parallel
    * `cudaEventSynchronize()` blocks all instruction processing until the GPU has reached the event
* `cudaEventElapsedTime(&f, start, stop)`: computes elapsed time (msec) between `start` and `stop`, stored as `float`

## Pinned memory
* CPU memory is _pageable_
    * can be swapped to disk
* pinned (page-locked) memory stays in place
* has performance advantage when copying to/from GPU due to "DMA" hardware speedup
    * more overhead when allocating pinned memory, so only useful if you're going to do lots of data transfer
* use `cudaHostAlloc()` instead of `malloc()` or `new` when allocating memory on the host side to allocate pinned memory
* use `cudaFreeHost()` to deallocate
* cannot be swapped out
    * must have enough
    * need to proactively deallocate in order to make sure there's enough
* really good lecture on the subject [here](https://kth.instructure.com/courses/12406/pages/optimizing-host-device-data-communication-i-pinned-host-memory)

## Streams
* remember `cudaEventRecord(event, stream)`?
* a CUDA _stream_ is a queue of GPU operations
    * kernel launch
    * memory copy
* streams allow a form of task-based parallelism
    * performance improvement if you use multiple streams
* to leverage streams you need device overlap support
    * `GPU_OVERLAP` needs to be enabled
* good tutorial [here](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)

# Stream API
* `cudaStream_t`: stream handler
* `cudaStreamCreate(&stream)`: creates/initializes stream
* `kernel<<<blocks,threads,shared,stream>>>`: add stream as 4th parameter to kernel call
* `cudaMemcpyAsync()`
    * normal `cudaMemcpy` is blocking; waits until operation is complete
    * this one is asynchronous, so it does not wait before moving on to next bit of code
    * must use pinned memory!
    * takes a stream parameter
    * no guarantee copy has completed when this function exits
* `stream` parameter
* `cudaStreamSynchronize(stream)`: similar to `cudaEventSynchronize()`