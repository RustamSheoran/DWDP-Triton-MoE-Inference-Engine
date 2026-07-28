#pragma once

#include <cuda_runtime_api.h>

#include <mutex>

namespace dwdp::communication {

class CUDAStreamPool final {
public:
  explicit CUDAStreamPool(int device_id = 0);
  ~CUDAStreamPool() noexcept;

  CUDAStreamPool(const CUDAStreamPool &) = delete;
  CUDAStreamPool &operator=(const CUDAStreamPool &) = delete;

  void initialize();
  void shutdown() noexcept;

  [[nodiscard]] cudaStream_t compute() const;
  [[nodiscard]] cudaStream_t copy() const;
  [[nodiscard]] int device() const noexcept;
  [[nodiscard]] bool initialized() const noexcept;

private:
  int device_id_;
  cudaStream_t compute_stream_{nullptr};
  cudaStream_t copy_stream_{nullptr};
  bool initialized_{false};
  mutable std::mutex mutex_;
};

} // namespace dwdp::communication
