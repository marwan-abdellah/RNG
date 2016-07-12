#include <cuda/Allocator.h>
#include <RandomState.h>

namespace CUDA
{

HOST template <class T>
Allocator<T>::Allocator()
{
    _hostData = nullptr;
    _deviceData = nullptr;
}

HOST template<class T>
void Allocator<T>::allocateMemory(size_t dataCount)
{
    if(dataCount < 1)
    {
        printf("ERROR ALLOCATING DATA \n");
        exit(EXIT_SUCCESS);
    }

    _maxDataCount = dataCount;

    // Allocate the host data
    _hostData = static_cast<T*>(malloc(sizeof(T) * dataCount));

    // Allocate the edevice data
    cudaMalloc(&_deviceData, sizeof(T) * dataCount);
}

HOST template<class T>
void Allocator<T>::uploadToGPU(T* hostData, size_t dataCount)
{
    /// Normally, the hostData pointer results from the data()
    /// function in the std::vector.
    // The hostData pointer should be valid
    // The hostData should be copied to _hostData and deleted from hostData later.
    // The dataCount should be less than the _maxDataCount

    if(dataCount <= _maxDataCount)
    memcpy(_hostData, hostData, sizeof(T) * dataCount);
    cudaMemcpy(_deviceData, _hostData,
               sizeof(T) * dataCount, cudaMemcpyHostToDevice);
}


HOST template<class T>
void Allocator<T>::downloadToCPU(size_t dataCount)
{
    if(dataCount <= _maxDataCount)
    {
        cudaMemcpy(_hostData, _deviceData, sizeof(T) * dataCount,
                   cudaMemcpyDeviceToHost);
    }
}

HOST template<class T>
T* Allocator<T>::hostData() const
{
    return _hostData;
}

HOST_DEVICE template<class T>
T* Allocator<T>::deviceData() const
{
    return _deviceData;
}

HOST template<class T>
void Allocator<T>::_release()
{
    if(_hostData != nullptr)
        free(_hostData);

    if(_deviceData != nullptr)
        cudaFree(_deviceData);

    _hostData = 0;
    _deviceData = 0;
}

HOST template<class T>
Allocator<T>::~Allocator()
{
    _release();
}

template class Allocator<RandomState>;
}
