#include "communication_engine.h"

#include <chrono>
#include <stdexcept>

namespace dwdp::communication {
CommunicationEngine::CommunicationEngine(int device_id) : streams_(device_id), events_(device_id), staging_(device_id) {}
CommunicationEngine::~CommunicationEngine() noexcept { shutdown(); }
void CommunicationEngine::initialize(std::size_t staging_bytes) { std::scoped_lock lock(mutex_); if (initialized_) return; streams_.initialize(); try { events_.initialize(1); staging_.allocate(staging_bytes); worker_ = std::make_unique<PrefetchWorker>(weights_, staging_, events_, prefetch_queue_, streams_); worker_->start(); initialized_ = true; } catch (...) { worker_.reset(); staging_.free(); events_.shutdown(); streams_.shutdown(); throw; } }
void CommunicationEngine::shutdown() noexcept { std::scoped_lock lock(mutex_); if (!initialized_) return; worker_->stop(); worker_.reset(); ipc_.closeAll(); weights_.clear(); staging_.free(); events_.shutdown(); streams_.shutdown(); initialized_ = false; }
void CommunicationEngine::registerExpert(int expert_id, void* source_device_pointer, std::size_t size_bytes) { std::scoped_lock lock(mutex_); if (!initialized_) throw std::logic_error("CommunicationEngine is not initialized"); if (size_bytes > staging_.capacity()) throw std::out_of_range("expert exceeds staging capacity"); weights_.registerExpert(expert_id, source_device_pointer, size_bytes, events_.acquire()); }
void CommunicationEngine::prefetch(int expert_id) { if (!initialized_) throw std::logic_error("CommunicationEngine is not initialized"); prefetch_queue_.enqueue(PrefetchRequest{expert_id}); }
void CommunicationEngine::wait(int expert_id) { std::scoped_lock lock(mutex_); events_.wait(weights_.copyEventIndex(expert_id), streams_.compute()); }
void* CommunicationEngine::getWeight(int expert_id) const { return weights_.getDevicePointer(expert_id); }
bool CommunicationEngine::isResident(int expert_id) const { const auto record = weights_.getRecord(expert_id); return record.resident && record.buffer_index == staging_.currentIndex(); }
void* CommunicationEngine::getResidentPointer(int expert_id) { const auto record = weights_.waitForResident(expert_id); events_.wait(record.copy_event_index, streams_.compute()); if (record.buffer_index != staging_.currentIndex()) throw std::logic_error("expert is resident in next buffer; call swapBuffers after prior compute"); weights_.markAccessed(expert_id, static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())); return weights_.getResidentPointer(expert_id); }
void CommunicationEngine::swapBuffers() { std::scoped_lock lock(mutex_); staging_.swap(); worker_->notifyBufferAvailable(); }
WeightManager& CommunicationEngine::weights() noexcept { return weights_; }
const WeightManager& CommunicationEngine::weights() const noexcept { return weights_; }
bool CommunicationEngine::initialized() const noexcept { return initialized_; }
}  // namespace dwdp::communication
