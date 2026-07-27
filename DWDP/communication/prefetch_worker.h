#pragma once

#include "buffers.h"
#include "cache_manager.h"
#include "events.h"
#include "ipc_manager.h"
#include "prefetch_queue.h"
#include "streams.h"
#include "weight_manager.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace dwdp::communication {
class PrefetchWorker final {
 public:
  PrefetchWorker(WeightManager&, CacheManager&, IPCManager&, DoubleBufferedStaging&, CUDAEventPool&, PrefetchQueue&, CUDAStreamPool&);
  ~PrefetchWorker() noexcept;
  void start();
  void stop() noexcept;
  void notifyBufferAvailable();
 private:
  void run() noexcept;
  WeightManager& weights_; CacheManager& cache_; IPCManager& ipc_; DoubleBufferedStaging& staging_; CUDAEventPool& events_; PrefetchQueue& queue_; CUDAStreamPool& streams_;
  std::thread thread_; std::atomic<bool> running_{false}; std::mutex mutex_; std::condition_variable ready_; bool buffer_available_{true};
};
}  // namespace dwdp::communication
