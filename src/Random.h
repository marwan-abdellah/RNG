#ifndef RANDOM_H
#define RANDOM_H

#include <RandomState.h>

class Random
{
public:

    HOST_DEVICE
    Random()
    {
        _state.state = 0x853c49e6748fea9bULL;
        _state.increment = 0xda3e39cb94b95bdbULL;
    }

    HOST_DEVICE
    explicit Random(uint64_t seed0)
    {
        seed(seed0);
    }

    HOST_DEVICE
    explicit Random(RandomState state0)
    {
        seed(state0);
    }

public:

    HOST_DEVICE
    void seed(uint64_t seed1)
    {
        _state.state = seed1;
        _state.increment = reinterpret_cast<uint64_t>(this);
    }

    HOST_DEVICE
    void seed(RandomState state1)
    {
        _state = state1;
    }

    HOST_DEVICE
    static uint32_t min() { return 0; }

    HOST_DEVICE
    static uint32_t max() { return UINT32_MAX; }

    HOST_DEVICE
    uint32_t operator()()
    {
        uint64_t prevState = _state.state;
        _state.state = prevState * 6364136223846793005ULL + _state.increment;
        uint32_t xorshifted =
                static_cast<uint32_t>(((prevState >> 18u) ^ prevState) >> 27u);
        uint32_t rot = static_cast<uint32_t>(prevState >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

public:
    RandomState _state;

};

#endif // RANDOM_H
