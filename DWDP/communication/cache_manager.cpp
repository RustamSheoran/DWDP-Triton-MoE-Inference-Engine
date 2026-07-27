#include "cache_manager.h"

#include <stdexcept>

namespace dwdp::communication {
CacheManager::CacheManager(std::size_t capacity_bytes) : capacity_bytes_(capacity_bytes) { if (capacity_bytes == 0) throw std::invalid_argument("cache capacity must be non-zero"); }
bool CacheManager::contains(int id) const { std::scoped_lock lock(mutex_); return entries_.find(id) != entries_.end(); }
std::size_t CacheManager::capacity() const noexcept { return capacity_bytes_; }
std::size_t CacheManager::used() const noexcept { std::scoped_lock lock(mutex_); return used_bytes_; }
std::size_t CacheManager::freeBytes() const noexcept { std::scoped_lock lock(mutex_); return capacity_bytes_ - used_bytes_; }
std::vector<int> CacheManager::admit(int id, std::size_t bytes) { if (bytes > capacity_bytes_) throw std::out_of_range("expert exceeds cache capacity"); std::scoped_lock lock(mutex_); const auto existing = entries_.find(id); if (existing != entries_.end()) { touchLocked(existing); return {}; } std::vector<int> evicted; while (used_bytes_ + bytes > capacity_bytes_) { auto victim = lru_.begin(); while (victim != lru_.end() && entries_.at(*victim).pins != 0) ++victim; if (victim == lru_.end()) throw std::runtime_error("cache is full of pinned experts"); const int victim_id = *victim; used_bytes_ -= entries_.at(victim_id).bytes; entries_.erase(victim_id); lru_.erase(victim); evicted.push_back(victim_id); } lru_.push_back(id); entries_.emplace(id, Entry{bytes, 0, std::prev(lru_.end())}); used_bytes_ += bytes; return evicted; }
void CacheManager::touchLocked(std::unordered_map<int, Entry>::iterator entry) { lru_.splice(lru_.end(), lru_, entry->second.lru); entry->second.lru = std::prev(lru_.end()); }
void CacheManager::touch(int id) { std::scoped_lock lock(mutex_); const auto it = entries_.find(id); if (it == entries_.end()) throw std::out_of_range("cache entry missing"); touchLocked(it); }
void CacheManager::pin(int id) { std::scoped_lock lock(mutex_); const auto it = entries_.find(id); if (it == entries_.end()) throw std::out_of_range("cache entry missing"); ++it->second.pins; touchLocked(it); }
void CacheManager::unpin(int id) { std::scoped_lock lock(mutex_); const auto it = entries_.find(id); if (it == entries_.end()) throw std::out_of_range("cache entry missing"); if (it->second.pins == 0) throw std::logic_error("cache pin underflow"); --it->second.pins; }
void CacheManager::erase(int id) { std::scoped_lock lock(mutex_); const auto it = entries_.find(id); if (it == entries_.end()) return; if (it->second.pins != 0) throw std::logic_error("cannot erase pinned cache entry"); used_bytes_ -= it->second.bytes; lru_.erase(it->second.lru); entries_.erase(it); }
}  // namespace dwdp::communication
