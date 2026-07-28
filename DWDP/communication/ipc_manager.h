#pragma once

#include "ipc.h"

#include <mutex>
#include <unordered_map>

namespace dwdp::communication {

class IPCManager final {
public:
  ~IPCManager() noexcept;
  cudaIpcMemHandle_t exportExpert(const void *pointer) const;
  void *importExpert(int expert_id, const cudaIpcMemHandle_t &handle);
  void closeImported(int expert_id);
  void closeAll() noexcept;

private:
  std::mutex mutex_;

  struct ImportedMapping {
    void *pointer;
    std::size_t references;
  };

  std::unordered_map<int, ImportedMapping> imported_;
};

} // namespace dwdp::communication
