#include "streams.h"

#include "cuda_check.h"

#include <stdexcept>

namespace dwdp::communication {

CUDAStreamPool::CUDAStreamPool(int device_id) : device_id_(device_id) {
  if (device_id < 0) {
    throw std::invalid_argument("device_id must be non-negative");
  }
}

CUDAStreamPool::~CUDAStreamPool() noexcept {
  shutdown();
}

void CUDAStreamPool::initialize() {
  std::scoped_lock lock(mutex_);
  if (initialized_) {
    return;
  }

  int previous_device = 0;
  DWDP_CUDA_CHECK(cudaGetDevice(&previous_device));
  DWDP_CUDA_CHECK(cudaSetDevice(device_id_));
  try {
    DWDP_CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking));
    DWDP_CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    initialized_ = true;
  } catch (...) {
    if (copy_stream_ != nullptr) {
      cudaStreamDestroy(copy_stream_);
      copy_stream_ = nullptr;
    }
    if (compute_stream_ != nullptr) {
      cudaStreamDestroy(compute_stream_);
      compute_stream_ = nullptr;
    }
    cudaSetDevice(previous_device);
    throw;
  }
  DWDP_CUDA_CHECK(cudaSetDevice(previous_device));
}

void CUDAStreamPool::shutdown() noexcept {
  std::scoped_lock lock(mutex_);
  if (!initialized_) {
    return;
  }
  const int previous_device = [] {
    int device = 0;
    cudaGetDevice(&device);
    return device;
  }();
  cudaSetDevice(device_id_);
  if (copy_stream_ != nullptr) {
    cudaStreamDestroy(copy_stream_);
    copy_stream_ = nullptr;
  }
  if (compute_stream_ != nullptr) {
    cudaStreamDestroy(compute_stream_);
    compute_stream_ = nullptr;
  }
  cudaSetDevice(previous_device);
  initialized_ = false;
}

cudaStream_t CUDAStreamPool::compute() const {
  std::scoped_lock lock(mutex_);
  if (!initialized_) {
    throw std::logic_error("CUDAStreamPool is not initialized");
  }
  return compute_stream_;
}

cudaStream_t CUDAStreamPool::copy() const {
  std::scoped_lock lock(mutex_);
  if (!initialized_) {
    throw std::logic_error("CUDAStreamPool is not initialized");
  }
  return copy_stream_;
}

int CUDAStreamPool::device() const noexcept {
  return device_id_;
}
bool CUDAStreamPool::initialized() const noexcept {
  return initialized_;
}

} // namespace dwdp::communication
