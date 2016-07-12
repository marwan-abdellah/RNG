#ifndef SHARED_HH
#define SHARED_HH

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

#define HOST            __host__
#define DEVICE          __device__
#define HOST_DEVICE     __host__ __device__

#endif // SHARED_HH
