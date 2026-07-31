#pragma once

#include <cuda_runtime_api.h>

namespace dwdp::communication::ipc {

cudaIpcMemHandle_t exportHandle(const void* device_pointer);
void* openHandle(const cudaIpcMemHandle_t& handle,
                 unsigned int flags = cudaIpcMemLazyEnablePeerAccess);
void closeHandle(void* device_pointer);

}  // namespace dwdp::communication::ipc
