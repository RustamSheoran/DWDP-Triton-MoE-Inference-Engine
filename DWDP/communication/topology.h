#pragma once

#include <mutex>
#include <unordered_map>

namespace dwdp::communication {
class PeerTopology final {
 public:
  [[nodiscard]] bool canAccess(int source_gpu, int destination_gpu);
 private:
  [[nodiscard]] static long long key(int source_gpu, int destination_gpu);
  std::mutex mutex_;
  std::unordered_map<long long, bool> cache_;
};
}  // namespace dwdp::communication
