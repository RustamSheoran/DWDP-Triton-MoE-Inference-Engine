#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace dwdp::communication {

enum class BufferLocation : std::uint8_t { kResident = 0, kStagingA = 1, kStagingB = 2, kImportedIPC = 3 };

struct ExpertRecord {
  void* device_pointer{nullptr};
  std::size_t size_bytes{0};
  BufferLocation location{BufferLocation::kResident};
  cudaIpcMemHandle_t ipc_handle{};
  bool has_ipc_handle{false};
  bool resident{false};
};

class WeightManager final {
 public:
  WeightManager() = default;
  ~WeightManager() = default;

  WeightManager(const WeightManager&) = delete;
  WeightManager& operator=(const WeightManager&) = delete;

  void registerExpert(int expert_id, void* device_pointer, std::size_t size_bytes,
                      BufferLocation location = BufferLocation::kResident,
                      const cudaIpcMemHandle_t* ipc_handle = nullptr);
  void unregisterExpert(int expert_id);
  void prefetchAsync(int expert_id, void* destination, std::size_t destination_bytes,
                     cudaStream_t copy_stream) const;
  [[nodiscard]] void* getDevicePointer(int expert_id) const;
  [[nodiscard]] bool contains(int expert_id) const;
  void clear();

 private:
  std::unordered_map<int, ExpertRecord> records_;
  mutable std::mutex mutex_;
};

}  // namespace dwdp::communication
