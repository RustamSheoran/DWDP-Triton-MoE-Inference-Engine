#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace dwdp::communication {

class CacheManager final {
 public:
  explicit CacheManager(std::size_t capacity_bytes);
  [[nodiscard]] bool contains(int expert_id) const;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t used() const noexcept;
  [[nodiscard]] std::size_t freeBytes() const noexcept;
  [[nodiscard]] std::vector<int> admit(int expert_id, std::size_t bytes);
  void touch(int expert_id);
  void pin(int expert_id);
  void unpin(int expert_id);
  void erase(int expert_id);

 private:
  struct Entry {
    std::size_t bytes;
    std::size_t pins;
    std::list<int>::iterator lru;
  };

  void touchLocked(std::unordered_map<int, Entry>::iterator entry);
  std::size_t capacity_bytes_;
  std::size_t used_bytes_{0};
  std::list<int> lru_;
  std::unordered_map<int, Entry> entries_;
  mutable std::mutex mutex_;
};

}  // namespace dwdp::communication
