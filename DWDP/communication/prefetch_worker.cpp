#include "prefetch_worker.h"

namespace dwdp::communication {

PrefetchWorker::PrefetchWorker(WeightManager& w, CacheManager& c, IPCManager& i,
                               DoubleBufferedStaging& b, CUDAEventPool& e, PrefetchQueue& q,
                               CUDAStreamPool& s)
    : weights_(w), cache_(c), ipc_(i), staging_(b), events_(e), queue_(q), streams_(s) {
}

PrefetchWorker::~PrefetchWorker() noexcept {
  stop();
}

void PrefetchWorker::start() {
  if (!running_.exchange(true)) {
    thread_ = std::thread(&PrefetchWorker::run, this);
  }
}

void PrefetchWorker::stop() noexcept {
  if (running_.exchange(false)) {
    queue_.close();
    ready_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }
}

void PrefetchWorker::notifyBufferAvailable() {
  std::scoped_lock lock(mutex_);
  buffer_available_ = true;
  ready_.notify_one();
}

void PrefetchWorker::run() noexcept {
  while (running_) {
    const auto request = queue_.waitDequeue();
    if (!request) {
      return;
    }
    try {
      const auto present = weights_.getRecord(request->expert_id);
      if (present.state == ResidentState::kActive || present.state == ResidentState::kStaged) {
        continue;
      }
      if (present.has_ipc_handle) {
        void* pointer = ipc_.importExpert(request->expert_id, present.ipc_handle);
        weights_.publishIPC(request->expert_id, pointer);
        cache_.admit(request->expert_id, present.size_bytes);
        continue;
      }
      std::unique_lock lock(mutex_);
      ready_.wait(lock, [this] {
        return !running_ || buffer_available_;
      });
      if (!running_) {
        return;
      }
      buffer_available_ = false;
      lock.unlock();
      const auto record = weights_.beginLoad(request->expert_id);
      const auto index = staging_.nextIndex();
      void* destination = staging_.next();
      staging_.copyToNextAsync(record.device_pointer, record.size_bytes, cudaMemcpyDeviceToDevice,
                               streams_.copy());
      events_.record(record.copy_event_index, streams_.copy());
      weights_.completeLoad(request->expert_id, destination, index);
      for (const int victim : cache_.admit(request->expert_id, record.size_bytes)) {
        weights_.evict(victim);
      }
    } catch (...) {
      weights_.failLoad(request->expert_id);
      notifyBufferAvailable();
    }
  }
}

}  // namespace dwdp::communication
