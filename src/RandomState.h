#ifndef RANDOMSTATE_H
#define RANDOMSTATE_H

#include <Shared.hh>

struct RandomState
{
    uint64_t state;
    uint64_t increment;
};

#endif // RANDOMSTATE_H
