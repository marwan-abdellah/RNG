#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <Shared.hh>

namespace CUDA
{
template <typename T>
class Allocator
{
public:
    Allocator();
    ~Allocator();

public:
    void allocateMemory(size_t count);
    void uploadToGPU(T* hostData, size_t dataCount);
    void downloadToCPU(size_t dataCount);

    HOST_DEVICE
    T* deviceData() const;
    T* hostData() const;

private:
    void _release();

private:
    T* _hostData;
    T* _deviceData;
    size_t _maxDataCount;
};
}

#endif // ALLOCATOR_H
