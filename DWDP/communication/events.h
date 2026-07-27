#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <mutex>
#include <vector>

namespace dwdp::communication {

class CUDAEventPool final {
 public:
  explicit CUDAEventPool(int device_id = 0);
  ~CUDAEventPool() noexcept;

  CUDAEventPool(const CUDAEventPool&) = delete;
  CUDAEventPool& operator=(const CUDAEventPool&) = delete;

  void initialize(std::size_t count);
  void shutdown() noexcept;
  void record(std::size_t index, cudaStream_t stream);
  void wait(std::size_t index, cudaStream_t stream) const;
  [[nodiscard]] cudaEvent_t event(std::size_t index) const;
  [[nodiscard]] std::size_t size() const noexcept;

 private:
  void validateIndex(std::size_t index) const;

  int device_id_;
  std::vector<cudaEvent_t> events_;
  mutable std::mutex mutex_;
};

}  // namespace dwdp::communication
