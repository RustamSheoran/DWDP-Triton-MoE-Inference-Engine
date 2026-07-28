cmake_minimum_required(VERSION 3.18)
project(dwdp_communication LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_library(dwdp_communication STATIC
  buffers.cu
  cache_manager.cpp
  communication_policy.cpp
  communication_engine.cpp
  events.cu
  ipc.cu
  ipc_manager.cpp
  prefetch_queue.cpp
  prefetch_worker.cpp
  streams.cu
  topology.cpp
  transfer_scheduler.cpp
  weight_manager.cpp)

target_include_directories(dwdp_communication PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(dwdp_communication PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(dwdp_communication PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
