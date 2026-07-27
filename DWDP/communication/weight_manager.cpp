#include "weight_manager.h"

#include "cuda_check.h"

#include <stdexcept>

namespace dwdp::communication {

void WeightManager::registerExpert(int expert_id, void* device_pointer, std::size_t size_bytes,
                                   BufferLocation location, const cudaIpcMemHandle_t* ipc_handle) {
  if (expert_id < 0) throw std::invalid_argument("expert_id must be non-negative");
  if (device_pointer == nullptr) throw std::invalid_argument("expert device pointer must be non-null");
  if (size_bytes == 0) throw std::invalid_argument("expert size must be non-zero");
  std::scoped_lock lock(mutex_);
  ExpertRecord record;
  record.device_pointer = device_pointer;
  record.size_bytes = size_bytes;
  record.location = location;
  if (ipc_handle != nullptr) {
    record.ipc_handle = *ipc_handle;
    record.has_ipc_handle = true;
  }
  record.resident = true;
  const auto inserted = records_.emplace(expert_id, record).second;
  if (!inserted) throw std::logic_error("expert is already registered");
}

void WeightManager::unregisterExpert(int expert_id) {
  std::scoped_lock lock(mutex_);
  if (records_.erase(expert_id) == 0) throw std::out_of_range("expert is not registered");
}

void WeightManager::prefetchAsync(int expert_id, void* destination, std::size_t destination_bytes,
                                  cudaStream_t copy_stream) const {
  if (destination == nullptr) throw std::invalid_argument("prefetch destination must be non-null");
  std::scoped_lock lock(mutex_);
  const auto it = records_.find(expert_id);
  if (it == records_.end()) throw std::out_of_range("expert is not registered");
  const ExpertRecord& record = it->second;
  if (!record.resident) throw std::logic_error("expert is not resident");
  if (destination_bytes < record.size_bytes) throw std::out_of_range("prefetch destination is too small");
  DWDP_CUDA_CHECK(cudaMemcpyAsync(destination, record.device_pointer, record.size_bytes,
                                   cudaMemcpyDeviceToDevice, copy_stream));
}

void* WeightManager::getDevicePointer(int expert_id) const {
  std::scoped_lock lock(mutex_);
  const auto it = records_.find(expert_id);
  if (it == records_.end()) throw std::out_of_range("expert is not registered");
  if (!it->second.resident) throw std::logic_error("expert is not resident");
  return it->second.device_pointer;
}

bool WeightManager::contains(int expert_id) const {
  std::scoped_lock lock(mutex_);
  return records_.find(expert_id) != records_.end();
}

void WeightManager::clear() {
  std::scoped_lock lock(mutex_);
  records_.clear();
}

}  // namespace dwdp::communication
