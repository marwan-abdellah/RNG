#ifndef KERNEL_CU
#define KERNEL_CU

#include <math.h>
#include <cuda_runtime.h>
#include <Kernel.h>
#include <RandomState.h>
#include <RNG.h>

// Declare the kernel function
__global__ void __kernel_fillArray(float* A, const int arraySize);
__global__ void __kernel_fillArrayWithRandomNumbers(float* A, const int arraySize, RandomState* states);


// function which invokes the kernel
void fillArray(float* A, const int arraySize)
{
    // Number of blocks per grid and the number of threads per block
    int threadsPerBlock, blocksPerGrid;

    threadsPerBlock = 512;
    blocksPerGrid   = ceil(double(arraySize)/double(threadsPerBlock));

    // invoke the kernel
    __kernel_fillArray <<< blocksPerGrid, threadsPerBlock >>>(A, arraySize);
}

void fillArrayWithRandomNumbers(float* A, const int arraySize, RandomState* states)
{
    // Number of blocks per grid and the number of threads per block
    int threadsPerBlock, blocksPerGrid;

    threadsPerBlock = 512;
    blocksPerGrid   = ceil(double(arraySize)/double(threadsPerBlock));

    // invoke the kernel
    __kernel_fillArrayWithRandomNumbers <<< blocksPerGrid, threadsPerBlock >>>(A, arraySize, states);
}

// Kernels
__global__ void __kernel_fillArray(float* A, const int arraySize)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId < arraySize)
        A[threadId] = 1.0;
}

__global__ void __kernel_fillArrayWithRandomNumbers(float* A, const int arraySize, RandomState* states)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    RNG rng(states[threadId]);

    if (threadId < arraySize)
        A[threadId] = rng.floatNumber();

    states[threadId] = rng.getState();
}

#endif // KERNEL_CU
