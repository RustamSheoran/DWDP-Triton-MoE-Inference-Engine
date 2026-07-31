#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <mutex>

namespace dwdp::communication {

class DoubleBufferedStaging final {
 public:
  explicit DoubleBufferedStaging(int device_id = 0);
  ~DoubleBufferedStaging() noexcept;

  DoubleBufferedStaging(const DoubleBufferedStaging&) = delete;
  DoubleBufferedStaging& operator=(const DoubleBufferedStaging&) = delete;

  void allocate(std::size_t bytes);
  void* current() const;
  void* next() const;
  void swap();
  void free() noexcept;
  void copyToNextAsync(const void* source, std::size_t bytes, cudaMemcpyKind kind,
                       cudaStream_t stream);
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t currentIndex() const noexcept;
  [[nodiscard]] std::size_t nextIndex() const noexcept;

 private:
  int device_id_;
  void* buffers_[2]{nullptr, nullptr};
  std::size_t bytes_{0};
  std::size_t current_index_{0};
  mutable std::mutex mutex_;
};

}  // namespace dwdp::communication
