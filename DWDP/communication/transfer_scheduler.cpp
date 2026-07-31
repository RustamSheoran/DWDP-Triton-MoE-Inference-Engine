#include "transfer_scheduler.h"

#include <stdexcept>

namespace dwdp::communication {

void TransferScheduler::transition(
    TransferTask& t, TransferState from, TransferState to) {
  if (t.state != from) {
    throw std::logic_error("illegal transfer transition");
  }
  t.state = to;
}

std::shared_ptr<TransferTask> TransferScheduler::submit(
    int id, int priority, std::function<void(TransferState)> cb) {
  if (id < 0) {
    throw std::invalid_argument("invalid expert id");
  }
  std::scoped_lock lock(mutex_);
  if (closed_) {
    throw std::logic_error("scheduler closed");
  }
  if (const auto it = coalesced_.find(id); it != coalesced_.end()) {
    if (auto task = it->second.lock()) {
      return task;
    }
  }
  auto task = std::make_shared<TransferTask>(TransferTask{
      id, priority, sequence_++, TransferState::kCreated, 0, std::move(cb)});
  transition(*task, TransferState::kCreated, TransferState::kQueued);
  coalesced_[id] = task;
  queue_.push(task);
  ready_.notify_one();
  return task;
}

std::optional<std::shared_ptr<TransferTask>> TransferScheduler::take() {
  std::unique_lock lock(mutex_);
  ready_.wait(lock, [&] {
    return closed_ || !queue_.empty();
  });
  if (queue_.empty()) {
    return std::nullopt;
  }
  auto task = queue_.top();
  queue_.pop();
  transition(*task, TransferState::kQueued, TransferState::kRunning);
  return task;
}

void TransferScheduler::complete(const std::shared_ptr<TransferTask>& t) {
  std::function<void(TransferState)> cb;
  {
    std::scoped_lock lock(mutex_);
    transition(*t, TransferState::kRunning, TransferState::kCompleted);
    coalesced_.erase(t->expert_id);
    cb = t->completion;
  }
  if (cb) {
    cb(TransferState::kCompleted);
  }
}

void TransferScheduler::fail(
    const std::shared_ptr<TransferTask>& t, bool retryable) {
  std::function<void(TransferState)> cb;
  {
    std::scoped_lock lock(mutex_);
    if (retryable && t->retries++ < 2) {
      transition(*t, TransferState::kRunning, TransferState::kQueued);
      queue_.push(t);
      ready_.notify_one();
      return;
    }
    transition(*t, TransferState::kRunning, TransferState::kFailed);
    coalesced_.erase(t->expert_id);
    cb = t->completion;
  }
  if (cb) {
    cb(TransferState::kFailed);
  }
}

void TransferScheduler::cancel(int id) {
  std::scoped_lock lock(mutex_);
  const auto it = coalesced_.find(id);
  if (it == coalesced_.end()) {
    return;
  }
  if (auto task = it->second.lock()) {
    if (task->state != TransferState::kQueued) {
      throw std::logic_error("only queued transfer may be cancelled");
    }
    task->state = TransferState::kCancelled;
  }
  coalesced_.erase(it);
}

void TransferScheduler::close() {
  std::scoped_lock lock(mutex_);
  closed_ = true;
  ready_.notify_all();
}

}  // namespace dwdp::communication
