#ifndef RNG_H
#define RNG_H

#include <Random.h>

class RNG
{
public:

    HOST_DEVICE
    RNG()
    {
        /// EMPTY CONSTRUCTOR
    }

    HOST_DEVICE
    explicit RNG(uint64_t seed0)
    {
        _randomGenerator.seed(seed0);
    }

    HOST_DEVICE
    explicit RNG(RandomState state0)
    {
        _randomGenerator.seed(state0);
    }

public:

    HOST_DEVICE
    void seed(uint64_t seed1)
    {
        _randomGenerator.seed(seed1);
    }

    HOST_DEVICE
    void seed(RandomState state1)
    {
        _randomGenerator.seed(state1);
    }

    HOST_DEVICE
    RandomState getState() const
    {
        return _randomGenerator._state;
    }

    HOST_DEVICE
    uint32_t uint32Number()
    {
        return _randomGenerator();
    }

    HOST_DEVICE
    uint32_t uint32Number(uint32_t max)
    {
        uint32_t threshold = -max % max;

        for (;;)
        {
            uint32_t value = _randomGenerator();

            if (value >= threshold)
                return value % max;
        }
    }

    HOST_DEVICE
    uint32_t uint32Number(uint32_t min, uint32_t max)
    {
        return uint32Number((max - min) + 1) + min;
    }

    HOST_DEVICE
    float floatNumber()
    {
        return float(_randomGenerator()) / float(0xFFFFFFFF);
    }

private:
    Random _randomGenerator;
};

#endif // RNG_H
