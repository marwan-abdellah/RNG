#ifndef KERNEL_H
#define KERNEL_H

#include <RandomState.h>

void fillArray(float* A, const int arraySize);
void fillArrayWithRandomNumbers(float* A, const int arraySize, RandomState* states);

#endif // KERNEL_H
