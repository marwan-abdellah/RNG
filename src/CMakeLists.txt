#####################################################################
### Copyright © 2015 - 2016
### Marwan Abdellah <abdellah.marwan@gmail.com>
#####################################################################

### Adding the source files to the CMake source tree
set( SOURCE_FILES
    RandomNumberGeneration.cpp
    cuda/Allocator.cpp
)

### Adding the header files to the CMake source tree
set( HEADER_FILES
    Shared.hh
    Kernel.h
    RandomState.h
    Random.h
    RNG.h
    cuda/Allocator.h
)

### Adding the kernels to the CMake source tree
set( KERNEL_FILES
    Kernel.cu
)

### Include the directories to the CMake source tree
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

### Generate the executable
cuda_add_executable( RNG
    ${SOURCE_FILES}
    ${HEADER_FILES}
    ${KERNEL_FILES}
)
