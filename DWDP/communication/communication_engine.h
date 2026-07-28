#pragma once

#include "buffers.h"
#include "cache_manager.h"
#include "events.h"
#include "ipc_manager.h"
#include "prefetch_queue.h"
#include "prefetch_worker.h"
#include "streams.h"
#include "weight_manager.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>

namespace dwdp::communication {

class CommunicationEngine final {
public:
  explicit CommunicationEngine(int device_id = 0);
  ~CommunicationEngine() noexcept;

  CommunicationEngine(const CommunicationEngine &) = delete;
  CommunicationEngine &operator=(const CommunicationEngine &) = delete;

  void initialize(std::size_t staging_bytes);
  void registerExpert(int expert_id, void *source_device_pointer, std::size_t size_bytes);
  void registerIPCExpert(int expert_id, const cudaIpcMemHandle_t &handle, std::size_t size_bytes);
  void shutdown() noexcept;
  void prefetch(int expert_id);
  void wait(int expert_id);
  [[nodiscard]] void *getWeight(int expert_id) const;
  [[nodiscard]] bool isResident(int expert_id) const;
  [[nodiscard]] void *getResidentPointer(int expert_id);
  void swapBuffers();
  void release(int expert_id);

  [[nodiscard]] WeightManager &weights() noexcept;
  [[nodiscard]] const WeightManager &weights() const noexcept;
  [[nodiscard]] bool initialized() const noexcept;

private:
  CUDAStreamPool streams_;
  CUDAEventPool events_;
  WeightManager weights_;
  std::unique_ptr<CacheManager> cache_;
  IPCManager ipc_;
  DoubleBufferedStaging staging_;
  PrefetchQueue prefetch_queue_;
  std::unique_ptr<PrefetchWorker> worker_;
  std::atomic<bool> initialized_{false};
  mutable std::mutex mutex_;
};

} // namespace dwdp::communication
