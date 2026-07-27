#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <unordered_map>

namespace dwdp::communication {

enum class BufferLocation : std::uint8_t { kResident = 0, kStagingA = 1, kStagingB = 2, kImportedIPC = 3 };

struct ExpertRecord {
  void* device_pointer{nullptr};
  void* staging_pointer{nullptr};
  std::size_t size_bytes{0};
  BufferLocation location{BufferLocation::kResident};
  cudaIpcMemHandle_t ipc_handle{};
  bool has_ipc_handle{false};
  bool ipc_imported{false};
  bool resident{false};
  bool loading{false};
  std::size_t buffer_index{0};
  std::size_t copy_event_index{0};
  std::uint64_t last_access_timestamp{0};
  std::size_t reference_count{0};
};

class WeightManager final {
 public:
  WeightManager() = default;
  ~WeightManager() = default;

  WeightManager(const WeightManager&) = delete;
  WeightManager& operator=(const WeightManager&) = delete;

  void registerExpert(int expert_id, void* device_pointer, std::size_t size_bytes,
                      std::size_t copy_event_index,
                      BufferLocation location = BufferLocation::kResident,
                      const cudaIpcMemHandle_t* ipc_handle = nullptr);
  void unregisterExpert(int expert_id);
  [[nodiscard]] ExpertRecord beginLoad(int expert_id);
  void completeLoad(int expert_id, void* staging_pointer, std::size_t buffer_index);
  void failLoad(int expert_id);
  [[nodiscard]] ExpertRecord getRecord(int expert_id) const;
  [[nodiscard]] ExpertRecord waitForResident(int expert_id) const;
  [[nodiscard]] std::size_t copyEventIndex(int expert_id) const;
  void markAccessed(int expert_id, std::uint64_t timestamp);
  void release(int expert_id);
  [[nodiscard]] void* getDevicePointer(int expert_id) const;
  [[nodiscard]] void* getResidentPointer(int expert_id) const;
  [[nodiscard]] bool contains(int expert_id) const;
  void clear();

 private:
  std::unordered_map<int, ExpertRecord> records_;
  mutable std::mutex mutex_;
  mutable std::condition_variable state_changed_;
};

}  // namespace dwdp::communication
