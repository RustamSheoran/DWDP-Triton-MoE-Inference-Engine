#include <stdexcept>

#include "cuda_check.h"
#include "ipc.h"

namespace dwdp::communication::ipc {

cudaIpcMemHandle_t exportHandle(const void* device_pointer) {
  if (device_pointer == nullptr) {
    throw std::invalid_argument("device pointer must be non-null");
  }
  cudaIpcMemHandle_t handle{};
  DWDP_CUDA_CHECK(
      cudaIpcGetMemHandle(&handle, const_cast<void*>(device_pointer)));
  return handle;
}

void* openHandle(const cudaIpcMemHandle_t& handle, unsigned int flags) {
  void* device_pointer = nullptr;
  DWDP_CUDA_CHECK(cudaIpcOpenMemHandle(&device_pointer, handle, flags));
  return device_pointer;
}

void closeHandle(void* device_pointer) {
  if (device_pointer == nullptr) {
    return;
  }
  DWDP_CUDA_CHECK(cudaIpcCloseMemHandle(device_pointer));
}

}  // namespace dwdp::communication::ipc
