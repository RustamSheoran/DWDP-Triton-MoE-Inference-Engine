#include "communication_engine.h"

#include <stdexcept>

namespace dwdp::communication {

CommunicationEngine::CommunicationEngine(int device_id)
    : streams_(device_id), events_(device_id), staging_(device_id) {}

CommunicationEngine::~CommunicationEngine() noexcept { shutdown(); }

void CommunicationEngine::initialize(std::size_t staging_bytes) {
  std::scoped_lock lock(mutex_);
  if (initialized_) return;
  streams_.initialize();
  try {
    events_.initialize(1);
    staging_.allocate(staging_bytes);
    initialized_ = true;
  } catch (...) {
    staging_.free();
    events_.shutdown();
    streams_.shutdown();
    throw;
  }
}

void CommunicationEngine::shutdown() noexcept {
  std::scoped_lock lock(mutex_);
  if (!initialized_) return;
  weights_.clear();
  staging_.free();
  events_.shutdown();
  streams_.shutdown();
  initialized_ = false;
}

void CommunicationEngine::prefetch(int expert_id) {
  std::scoped_lock lock(mutex_);
  if (!initialized_) throw std::logic_error("CommunicationEngine is not initialized");
  prefetch_queue_.enqueue(PrefetchRequest{expert_id});
  const auto request = prefetch_queue_.dequeue();
  if (!request.has_value()) throw std::logic_error("prefetch queue unexpectedly empty");
  weights_.prefetchAsync(request->expert_id, staging_.next(), staging_.capacity(), streams_.copy());
  events_.record(0, streams_.copy());
}

void CommunicationEngine::wait() {
  std::scoped_lock lock(mutex_);
  if (!initialized_) throw std::logic_error("CommunicationEngine is not initialized");
  events_.wait(0, streams_.compute());
}

void* CommunicationEngine::getWeight(int expert_id) const { return weights_.getDevicePointer(expert_id); }

void CommunicationEngine::swapBuffers() {
  std::scoped_lock lock(mutex_);
  if (!initialized_) throw std::logic_error("CommunicationEngine is not initialized");
  staging_.swap();
}

WeightManager& CommunicationEngine::weights() noexcept { return weights_; }
const WeightManager& CommunicationEngine::weights() const noexcept { return weights_; }
bool CommunicationEngine::initialized() const noexcept { return initialized_; }

}  // namespace dwdp::communication
