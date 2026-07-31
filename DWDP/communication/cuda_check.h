#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace dwdp::communication {

class CUDAError final : public std::runtime_error {
 public:
  CUDAError(
      cudaError_t error, const char* operation, const char* file, int line)
      : std::runtime_error(
            std::string(operation) + " failed at " + file + ":" +
            std::to_string(line) + ": " + cudaGetErrorString(error)),
        code_(error) {
  }

  [[nodiscard]] cudaError_t code() const noexcept {
    return code_;
  }

 private:
  cudaError_t code_;
};

inline void checkCuda(
    cudaError_t error, const char* operation, const char* file, int line) {
  if (error != cudaSuccess) {
    throw CUDAError(error, operation, file, line);
  }
}

}  // namespace dwdp::communication

#define DWDP_CUDA_CHECK(expression) \
  ::dwdp::communication::checkCuda( \
      (expression), #expression, __FILE__, __LINE__)
