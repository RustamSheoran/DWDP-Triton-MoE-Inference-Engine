#include "prefetch_queue.h"

#include <stdexcept>

namespace dwdp::communication {

void PrefetchQueue::enqueue(PrefetchRequest request) {
  if (request.expert_id < 0) throw std::invalid_argument("expert_id must be non-negative");
  std::scoped_lock lock(mutex_);
  queue_.push(request);
}

std::optional<PrefetchRequest> PrefetchQueue::dequeue() {
  std::scoped_lock lock(mutex_);
  if (queue_.empty()) return std::nullopt;
  const PrefetchRequest request = queue_.front();
  queue_.pop();
  return request;
}

std::optional<PrefetchRequest> PrefetchQueue::peek() const {
  std::scoped_lock lock(mutex_);
  if (queue_.empty()) return std::nullopt;
  return queue_.front();
}

bool PrefetchQueue::empty() const {
  std::scoped_lock lock(mutex_);
  return queue_.empty();
}

std::size_t PrefetchQueue::size() const {
  std::scoped_lock lock(mutex_);
  return queue_.size();
}

}  // namespace dwdp::communication
