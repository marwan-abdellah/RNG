#include <math.h>
#include <time.h>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <Kernel.h>
#include <vector>
#include <RandomState.h>
#include <cuda/Allocator.h>
#include <random>

// Declare the vectors' number of elements
#define ARRAY_SIZE 512

int main()
{
    // RandomState std::vector
    std::vector <RandomState> randomStates(ARRAY_SIZE);

    /// std::random_device is a uniformly-distributed integer random number
    /// generator that produces non-deterministic random numbers.
    std::random_device randomNumbers;

    /// A Mersenne Twister pseudo-random generator of 64-bit numbers with a
    /// state size of 19937 bits.
    std::mt19937_64 rng(randomNumbers());

    // Filling the _randomStates_ vector
    for(size_t i = 0; i < ARRAY_SIZE; i++)
    {
        randomStates.at(i).state = rng();
        randomStates.at(i).increment = rng();
    }

    // CUDA allocation and data transfer to the GPU
    CUDA::Allocator<RandomState> randomStatesVector;
    randomStatesVector.allocateMemory(ARRAY_SIZE);
    randomStatesVector.uploadToGPU(randomStates.data(), ARRAY_SIZE);

    // Host
    float* hostOutputArray = (float*) malloc(ARRAY_SIZE * sizeof(float));

    // Device
    float* deviceOutputArray;

    // initialize input vectors
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        hostOutputArray[i] = 0.f;
    }

    // Allocate device vectors in the device (GPU) memory
    cudaMalloc(&deviceOutputArray, ARRAY_SIZE * sizeof(float));

    // Copy input vectors from the host (CPU) memory to the device (GPU) memory
    cudaMemcpy(deviceOutputArray, hostOutputArray, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel
    // fillArray(deviceOutputArray, ARRAY_SIZE);
    fillArrayWithRandomNumbers(deviceOutputArray, ARRAY_SIZE, randomStatesVector.deviceData());

    // Results
    cudaMemcpy(hostOutputArray, deviceOutputArray, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < ARRAY_SIZE; i++)
        std::cout << hostOutputArray[i] << " ";
    std::cout << std::endl;

    // Free device memory
    cudaFree(deviceOutputArray);

    // Free host memory
    delete[] hostOutputArray;

    randomStatesVector.~Allocator();

    return cudaDeviceSynchronize();
}
