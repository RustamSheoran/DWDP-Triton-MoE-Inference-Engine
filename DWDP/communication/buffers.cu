#include <stdexcept>

#include "buffers.h"
#include "cuda_check.h"

namespace dwdp::communication {

DoubleBufferedStaging::DoubleBufferedStaging(int device_id)
    : device_id_(device_id) {
  if (device_id < 0) {
    throw std::invalid_argument("device_id must be non-negative");
  }
}

DoubleBufferedStaging::~DoubleBufferedStaging() noexcept {
  free();
}

void DoubleBufferedStaging::allocate(std::size_t bytes) {
  if (bytes == 0) {
    throw std::invalid_argument("staging allocation must be non-zero");
  }
  std::scoped_lock lock(mutex_);
  if (bytes_ == bytes && buffers_[0] != nullptr) {
    return;
  }
  if (buffers_[0] != nullptr) {
    throw std::logic_error("free staging buffers before resizing");
  }
  int previous_device = 0;
  DWDP_CUDA_CHECK(cudaGetDevice(&previous_device));
  DWDP_CUDA_CHECK(cudaSetDevice(device_id_));
  try {
    DWDP_CUDA_CHECK(cudaMalloc(&buffers_[0], bytes));
    DWDP_CUDA_CHECK(cudaMalloc(&buffers_[1], bytes));
    bytes_ = bytes;
    current_index_ = 0;
  } catch (...) {
    if (buffers_[1] != nullptr) {
      cudaFree(buffers_[1]);
    }
    if (buffers_[0] != nullptr) {
      cudaFree(buffers_[0]);
    }
    buffers_[0] = buffers_[1] = nullptr;
    cudaSetDevice(previous_device);
    throw;
  }
  DWDP_CUDA_CHECK(cudaSetDevice(previous_device));
}

void* DoubleBufferedStaging::current() const {
  std::scoped_lock lock(mutex_);
  if (buffers_[current_index_] == nullptr) {
    throw std::logic_error("staging buffers are not allocated");
  }
  return buffers_[current_index_];
}

void* DoubleBufferedStaging::next() const {
  std::scoped_lock lock(mutex_);
  const auto next_index = 1U - current_index_;
  if (buffers_[next_index] == nullptr) {
    throw std::logic_error("staging buffers are not allocated");
  }
  return buffers_[next_index];
}

void DoubleBufferedStaging::swap() {
  std::scoped_lock lock(mutex_);
  if (buffers_[0] == nullptr) {
    throw std::logic_error("staging buffers are not allocated");
  }
  current_index_ = 1U - current_index_;
}

void DoubleBufferedStaging::free() noexcept {
  std::scoped_lock lock(mutex_);
  int previous_device = 0;
  cudaGetDevice(&previous_device);
  cudaSetDevice(device_id_);
  for (auto& buffer : buffers_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
    }
    buffer = nullptr;
  }
  cudaSetDevice(previous_device);
  bytes_ = 0;
  current_index_ = 0;
}

void DoubleBufferedStaging::copyToNextAsync(
    const void* source,
    std::size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
  if (source == nullptr) {
    throw std::invalid_argument("copy source must be non-null");
  }
  std::scoped_lock lock(mutex_);
  if (bytes > bytes_) {
    throw std::out_of_range("copy exceeds staging capacity");
  }
  const auto next_index = 1U - current_index_;
  if (buffers_[next_index] == nullptr) {
    throw std::logic_error("staging buffers are not allocated");
  }
  DWDP_CUDA_CHECK(
      cudaMemcpyAsync(buffers_[next_index], source, bytes, kind, stream));
}

std::size_t DoubleBufferedStaging::capacity() const noexcept {
  std::scoped_lock lock(mutex_);
  return bytes_;
}

std::size_t DoubleBufferedStaging::currentIndex() const noexcept {
  std::scoped_lock lock(mutex_);
  return current_index_;
}

std::size_t DoubleBufferedStaging::nextIndex() const noexcept {
  std::scoped_lock lock(mutex_);
  return 1U - current_index_;
}

}  // namespace dwdp::communication
