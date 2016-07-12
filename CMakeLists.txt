#####################################################################
### Copyright © 2015 - 2016
### Marwan Abdellah <abdellah.marwan@gmail.com>
#####################################################################

cmake_minimum_required( VERSION 3.0 )

project( RNG )

### cmake modules
list( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )

#########################################################
#! Find packages
#########################################################
INCLUDE( CUDA )
INCLUDE( CUDA_SDK )

add_subdirectory( src )

