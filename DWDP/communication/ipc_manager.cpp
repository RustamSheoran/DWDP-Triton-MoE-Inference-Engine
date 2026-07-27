#include "ipc_manager.h"

#include <stdexcept>
namespace dwdp::communication {
IPCManager::~IPCManager() noexcept { closeAll(); }
cudaIpcMemHandle_t IPCManager::exportExpert(const void* pointer) const { return ipc::exportHandle(pointer); }
void* IPCManager::importExpert(int expert_id, const cudaIpcMemHandle_t& handle) { std::scoped_lock lock(mutex_); const auto it = imported_.find(expert_id); if (it != imported_.end()) return it->second; void* pointer = ipc::openHandle(handle); imported_.emplace(expert_id, pointer); return pointer; }
void IPCManager::closeImported(int expert_id) { std::scoped_lock lock(mutex_); const auto it = imported_.find(expert_id); if (it == imported_.end()) return; ipc::closeHandle(it->second); imported_.erase(it); }
void IPCManager::closeAll() noexcept { std::scoped_lock lock(mutex_); for (const auto& [_, pointer] : imported_) { if (pointer != nullptr) cudaIpcCloseMemHandle(pointer); } imported_.clear(); }
}  // namespace dwdp::communication
