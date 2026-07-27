#pragma once

#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace dwdp::communication {

struct PrefetchRequest {
  int expert_id;
};

class PrefetchQueue final {
 public:
  void enqueue(PrefetchRequest request);
  [[nodiscard]] std::optional<PrefetchRequest> dequeue();
  [[nodiscard]] std::optional<PrefetchRequest> waitDequeue();
  [[nodiscard]] std::optional<PrefetchRequest> peek() const;
  [[nodiscard]] bool empty() const;
  [[nodiscard]] std::size_t size() const;
  void close();

 private:
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::queue<PrefetchRequest> queue_;
  bool closed_{false};
};

}  // namespace dwdp::communication
