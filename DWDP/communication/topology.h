#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace dwdp::communication {

class PeerTopology final {
 public:
  [[nodiscard]] bool canAccess(int source_gpu, int destination_gpu);

 private:
  [[nodiscard]] static std::uint64_t key(int source_gpu, int destination_gpu);
  std::mutex mutex_;
  std::unordered_map<std::uint64_t, bool> cache_;
};

}  // namespace dwdp::communication
