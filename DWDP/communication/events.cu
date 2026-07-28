#include "events.h"

#include "cuda_check.h"

#include <stdexcept>

namespace dwdp::communication {

CUDAEventPool::CUDAEventPool(int device_id) : device_id_(device_id) {
  if (device_id < 0) {
    throw std::invalid_argument("device_id must be non-negative");
  }
}

CUDAEventPool::~CUDAEventPool() noexcept {
  shutdown();
}

void CUDAEventPool::initialize(std::size_t count) {
  if (count == 0) {
    throw std::invalid_argument("CUDAEventPool requires at least one event");
  }
  std::scoped_lock lock(mutex_);
  if (!events_.empty()) {
    if (events_.size() != count) {
      throw std::logic_error("CUDAEventPool cannot be resized after initialization");
    }
    return;
  }
  int previous_device = 0;
  DWDP_CUDA_CHECK(cudaGetDevice(&previous_device));
  DWDP_CUDA_CHECK(cudaSetDevice(device_id_));
  try {
    events_.resize(count, nullptr);
    for (auto &event : events_) {
      DWDP_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
  } catch (...) {
    for (auto &event : events_) {
      if (event != nullptr)
        cudaEventDestroy(event);
    }
    events_.clear();
    cudaSetDevice(previous_device);
    throw;
  }
  DWDP_CUDA_CHECK(cudaSetDevice(previous_device));
}

std::size_t CUDAEventPool::acquire() {
  std::scoped_lock lock(mutex_);
  int previous_device = 0;
  DWDP_CUDA_CHECK(cudaGetDevice(&previous_device));
  DWDP_CUDA_CHECK(cudaSetDevice(device_id_));
  cudaEvent_t event_handle = nullptr;
  try {
    DWDP_CUDA_CHECK(cudaEventCreateWithFlags(&event_handle, cudaEventDisableTiming));
    events_.push_back(event_handle);
  } catch (...) {
    if (event_handle != nullptr)
      cudaEventDestroy(event_handle);
    cudaSetDevice(previous_device);
    throw;
  }
  DWDP_CUDA_CHECK(cudaSetDevice(previous_device));
  return events_.size() - 1;
}

void CUDAEventPool::shutdown() noexcept {
  std::scoped_lock lock(mutex_);
  if (events_.empty())
    return;
  int previous_device = 0;
  cudaGetDevice(&previous_device);
  cudaSetDevice(device_id_);
  for (auto &event : events_) {
    if (event != nullptr)
      cudaEventDestroy(event);
  }
  events_.clear();
  cudaSetDevice(previous_device);
}

void CUDAEventPool::record(std::size_t index, cudaStream_t stream) {
  std::scoped_lock lock(mutex_);
  validateIndex(index);
  DWDP_CUDA_CHECK(cudaEventRecord(events_[index], stream));
}

void CUDAEventPool::wait(std::size_t index, cudaStream_t stream) const {
  std::scoped_lock lock(mutex_);
  validateIndex(index);
  DWDP_CUDA_CHECK(cudaStreamWaitEvent(stream, events_[index], 0));
}

cudaEvent_t CUDAEventPool::event(std::size_t index) const {
  std::scoped_lock lock(mutex_);
  validateIndex(index);
  return events_[index];
}

std::size_t CUDAEventPool::size() const noexcept {
  std::scoped_lock lock(mutex_);
  return events_.size();
}

void CUDAEventPool::validateIndex(std::size_t index) const {
  if (index >= events_.size())
    throw std::out_of_range("CUDA event index out of range");
}

} // namespace dwdp::communication
