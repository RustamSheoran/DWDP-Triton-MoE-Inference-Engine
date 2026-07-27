cmake_minimum_required(VERSION 3.18)
project(dwdp_communication LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(dwdp_communication STATIC
  buffers.cu
  communication_engine.cpp
  events.cu
  ipc.cu
  ipc_manager.cpp
  prefetch_queue.cpp
  prefetch_worker.cpp
  streams.cu
  weight_manager.cpp)

target_include_directories(dwdp_communication PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(dwdp_communication PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
