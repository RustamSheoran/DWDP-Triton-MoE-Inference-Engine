#include "prefetch_queue.h"

#include <condition_variable>
#include <stdexcept>

namespace dwdp::communication {

void PrefetchQueue::enqueue(PrefetchRequest request) {
  if (request.expert_id < 0)
    throw std::invalid_argument("expert_id must be non-negative");
  std::unique_lock lock(mutex_);
  if (closed_)
    throw std::logic_error("prefetch queue is closed");
  queue_.push(request);
  lock.unlock();
  ready_.notify_one();
}

std::optional<PrefetchRequest> PrefetchQueue::waitDequeue() {
  std::unique_lock lock(mutex_);
  ready_.wait(lock, [this] { return closed_ || !queue_.empty(); });
  if (queue_.empty())
    return std::nullopt;
  const PrefetchRequest request = queue_.front();
  queue_.pop();
  return request;
}

std::optional<PrefetchRequest> PrefetchQueue::dequeue() {
  std::scoped_lock lock(mutex_);
  if (queue_.empty())
    return std::nullopt;
  const PrefetchRequest request = queue_.front();
  queue_.pop();
  return request;
}

std::optional<PrefetchRequest> PrefetchQueue::peek() const {
  std::scoped_lock lock(mutex_);
  if (queue_.empty())
    return std::nullopt;
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

void PrefetchQueue::close() {
  std::scoped_lock lock(mutex_);
  closed_ = true;
  ready_.notify_all();
}

void PrefetchQueue::reset() {
  std::scoped_lock lock(mutex_);
  if (!queue_.empty())
    throw std::logic_error("cannot reset a non-empty prefetch queue");
  closed_ = false;
}

} // namespace dwdp::communication
