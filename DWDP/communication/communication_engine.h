#pragma once

#include "buffers.h"
#include "events.h"
#include "prefetch_queue.h"
#include "streams.h"
#include "weight_manager.h"

#include <cstddef>
#include <mutex>

namespace dwdp::communication {

class CommunicationEngine final {
 public:
  explicit CommunicationEngine(int device_id = 0);
  ~CommunicationEngine() noexcept;

  CommunicationEngine(const CommunicationEngine&) = delete;
  CommunicationEngine& operator=(const CommunicationEngine&) = delete;

  void initialize(std::size_t staging_bytes);
  void shutdown() noexcept;
  void prefetch(int expert_id);
  void wait();
  [[nodiscard]] void* getWeight(int expert_id) const;
  void swapBuffers();

  [[nodiscard]] WeightManager& weights() noexcept;
  [[nodiscard]] const WeightManager& weights() const noexcept;
  [[nodiscard]] bool initialized() const noexcept;

 private:
  CUDAStreamPool streams_;
  CUDAEventPool events_;
  WeightManager weights_;
  DoubleBufferedStaging staging_;
  PrefetchQueue prefetch_queue_;
  bool initialized_{false};
  mutable std::mutex mutex_;
};

}  // namespace dwdp::communication
