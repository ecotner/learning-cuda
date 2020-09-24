# Atomic operations

## Why atomics?
* consider the operation `x++`
    1. read `x` into a register
    2. increment register value
    3. write register back into x
    * effectively `{temp = x; temp = temp + 1; x = temp;}`
* if two threads do `x++`
    * each thread has its own version of `temp`
    * we get a race condition; the second thread overwrites the results of the first one

## Atomic functions
* problem many threads accessing the same memory location
* atomic operations ensure that only one thread can access the location
    * comes with a performance cost
* they actually have grid scope! (surprising)
* `atomicOP(x, y)`
    * `t1 = *x;` (read)
    * `t2 = t2 OP y;` (modify)
    * `*a = t2;` (write)
* atomics need to be configured
    * `#include "sm_20_atomic_functions.h"`
    * have to specify GPU architecture

## Monte Carlo $\pi$
* want to evaluate $\pi = 3.141592...$ numerically
* some mathematical facts:
    * area of a circle $A_c = \pi r^2$
    * area of square enclosing the circle $A_s = 4r^2$
    * ratio of areas $A_c/A_s = \pi/4$
    * all points within a circle satisfy $x^2 + y^2 \le r^2$
* if we generate uniform random values within a unit square, then the fraction of points that fall within the unit circle is $\pi/4$
    * $\pi = \#(\text{points in circle})/\#(\text{all points})$