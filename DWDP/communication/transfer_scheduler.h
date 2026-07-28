#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace dwdp::communication {

enum class TransferState : std::uint8_t {
  kCreated,
  kQueued,
  kRunning,
  kWaiting,
  kCompleted,
  kFailed,
  kCancelled
};

struct TransferTask {
  int expert_id;
  int priority;
  std::uint64_t sequence;
  TransferState state{TransferState::kCreated};
  std::size_t retries{0};
  std::function<void(TransferState)> completion;
};

class TransferScheduler final {
public:
  std::shared_ptr<TransferTask>
  submit(int expert_id, int priority,
         std::function<void(TransferState)> completion = {});
  std::optional<std::shared_ptr<TransferTask>> take();
  void complete(const std::shared_ptr<TransferTask> &task);
  void fail(const std::shared_ptr<TransferTask> &task, bool retryable);
  void cancel(int expert_id);
  void close();

private:
  struct Order {
    bool operator()(const std::shared_ptr<TransferTask> &a,
                    const std::shared_ptr<TransferTask> &b) const {
      return a->priority == b->priority ? a->sequence > b->sequence
                                        : a->priority < b->priority;
    }
  };

  void transition(TransferTask &task, TransferState from, TransferState to);
  std::mutex mutex_;
  std::condition_variable ready_;
  std::priority_queue<std::shared_ptr<TransferTask>,
                      std::vector<std::shared_ptr<TransferTask>>, Order>
      queue_;
  std::unordered_map<int, std::weak_ptr<TransferTask>> coalesced_;
  std::uint64_t sequence_{0};
  bool closed_{false};
};

} // namespace dwdp::communication
