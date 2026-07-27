#include "weight_manager.h"

#include <stdexcept>

namespace dwdp::communication {

void WeightManager::registerExpert(int expert_id, void* device_pointer, std::size_t size_bytes,
                                   std::size_t copy_event_index, BufferLocation location,
                                   const cudaIpcMemHandle_t* ipc_handle) {
  if (expert_id < 0 || size_bytes == 0 || (device_pointer == nullptr && ipc_handle == nullptr)) {
    throw std::invalid_argument("invalid expert registration");
  }
  ExpertRecord record;
  record.expert_id = expert_id;
  record.device_pointer = device_pointer;
  record.size_bytes = size_bytes;
  record.location = location;
  record.copy_event_index = copy_event_index;
  if (ipc_handle != nullptr) {
    record.ipc_handle = *ipc_handle;
    record.has_ipc_handle = true;
  }
  std::scoped_lock lock(mutex_);
  if (!records_.emplace(expert_id, record).second) throw std::logic_error("expert is already registered");
}

void WeightManager::unregisterExpert(int expert_id) { std::scoped_lock lock(mutex_); if (records_.erase(expert_id) == 0) throw std::out_of_range("expert is not registered"); }

ExpertRecord WeightManager::beginLoad(int expert_id) {
  std::scoped_lock lock(mutex_);
  auto it = records_.find(expert_id);
  if (it == records_.end()) throw std::out_of_range("expert is not registered");
  if (it->second.state != ResidentState::kUnloaded && it->second.state != ResidentState::kEvicted) throw std::logic_error("illegal transition to loading");
  it->second.loading = true;
  it->second.state = ResidentState::kLoading;
  return it->second;
}

void WeightManager::completeLoad(int expert_id, void* staging_pointer, std::size_t buffer_index) {
  if (staging_pointer == nullptr || buffer_index > 1) throw std::invalid_argument("invalid staging completion");
  std::scoped_lock lock(mutex_);
  auto it = records_.find(expert_id);
  if (it == records_.end()) throw std::out_of_range("expert is not registered");
  it->second.staging_pointer = staging_pointer;
  it->second.buffer_index = buffer_index;
  it->second.location = buffer_index == 0 ? BufferLocation::kStagingA : BufferLocation::kStagingB;
  if (it->second.state != ResidentState::kLoading) throw std::logic_error("illegal transition to staged");
  it->second.resident = true;
  it->second.loading = false;
  it->second.resident_pointer = staging_pointer;
  it->second.state = ResidentState::kStaged;
  state_changed_.notify_all();
}

void WeightManager::activate(int expert_id, std::size_t active_buffer) { std::scoped_lock lock(mutex_); auto it = records_.find(expert_id); if (it == records_.end()) throw std::out_of_range("expert is not registered"); if (it->second.state != ResidentState::kStaged || it->second.buffer_index != active_buffer) throw std::logic_error("illegal transition to active"); it->second.state = ResidentState::kActive; }
void WeightManager::evict(int expert_id) { std::scoped_lock lock(mutex_); auto it = records_.find(expert_id); if (it == records_.end()) return; if (it->second.reference_count != 0) throw std::logic_error("cannot evict referenced expert"); if (it->second.state != ResidentState::kStaged && it->second.state != ResidentState::kActive) throw std::logic_error("illegal transition to evicted"); it->second.resident = false; it->second.resident_pointer = nullptr; it->second.staging_pointer = nullptr; it->second.state = ResidentState::kEvicted; }
void WeightManager::publishIPC(int expert_id, void* ipc_pointer) { if (ipc_pointer == nullptr) throw std::invalid_argument("IPC pointer must be non-null"); std::scoped_lock lock(mutex_); auto it = records_.find(expert_id); if (it == records_.end()) throw std::out_of_range("expert is not registered"); if (it->second.state != ResidentState::kUnloaded && it->second.state != ResidentState::kEvicted) throw std::logic_error("illegal IPC publication"); it->second.ipc_pointer = ipc_pointer; it->second.resident_pointer = ipc_pointer; it->second.ipc_imported = true; it->second.resident = true; it->second.state = ResidentState::kActive; state_changed_.notify_all(); }
std::vector<int> WeightManager::invalidateBuffer(std::size_t buffer_index) { std::scoped_lock lock(mutex_); std::vector<int> ids; for (auto& [id, record] : records_) { if (record.ipc_imported || record.buffer_index != buffer_index || (record.state != ResidentState::kStaged && record.state != ResidentState::kActive)) continue; if (record.reference_count != 0) throw std::logic_error("cannot recycle a referenced staging buffer"); record.resident = false; record.resident_pointer = nullptr; record.staging_pointer = nullptr; record.state = ResidentState::kEvicted; ids.push_back(id); } return ids; }

void WeightManager::failLoad(int expert_id) { std::scoped_lock lock(mutex_); auto it = records_.find(expert_id); if (it == records_.end()) return; it->second.loading = false; state_changed_.notify_all(); }

ExpertRecord WeightManager::getRecord(int expert_id) const { std::scoped_lock lock(mutex_); const auto it = records_.find(expert_id); if (it == records_.end()) throw std::out_of_range("expert is not registered"); return it->second; }
ExpertRecord WeightManager::waitForResident(int expert_id) const { std::unique_lock lock(mutex_); state_changed_.wait(lock, [&] { const auto it = records_.find(expert_id); return it == records_.end() || it->second.resident || !it->second.loading; }); const auto it = records_.find(expert_id); if (it == records_.end()) throw std::out_of_range("expert is not registered"); if (!it->second.resident) throw std::runtime_error("expert prefetch failed or was not scheduled"); return it->second; }
std::size_t WeightManager::copyEventIndex(int expert_id) const { return getRecord(expert_id).copy_event_index; }
void WeightManager::markAccessed(int expert_id, std::uint64_t timestamp) { std::scoped_lock lock(mutex_); auto it = records_.find(expert_id); if (it == records_.end()) throw std::out_of_range("expert is not registered"); ++it->second.reference_count; it->second.last_access_timestamp = timestamp; }
void WeightManager::release(int expert_id) { std::scoped_lock lock(mutex_); auto it = records_.find(expert_id); if (it == records_.end()) throw std::out_of_range("expert is not registered"); if (it->second.reference_count != 0) --it->second.reference_count; }
void* WeightManager::getDevicePointer(int expert_id) const { return getRecord(expert_id).device_pointer; }
void* WeightManager::getResidentPointer(int expert_id) const { const auto record = getRecord(expert_id); if (!record.resident || record.resident_pointer == nullptr || (record.state != ResidentState::kStaged && record.state != ResidentState::kActive)) throw std::logic_error("expert is not resident"); return record.resident_pointer; }
bool WeightManager::contains(int expert_id) const { std::scoped_lock lock(mutex_); return records_.find(expert_id) != records_.end(); }
void WeightManager::clear() { std::scoped_lock lock(mutex_); records_.clear(); }

}  // namespace dwdp::communication
