#include "topology.h"

#include "cuda_check.h"

#include <stdexcept>

namespace dwdp::communication {

std::uint64_t PeerTopology::key(int source, int destination) {
  const auto source_word = static_cast<std::uint32_t>(source);
  const auto destination_word = static_cast<std::uint32_t>(destination);
  const auto source_bits = static_cast<std::uint64_t>(source_word);
  const auto destination_bits = static_cast<std::uint64_t>(destination_word);
  return (source_bits << 32U) | destination_bits;
}

bool PeerTopology::canAccess(int source, int destination) {
  if (source < 0 || destination < 0) {
    throw std::invalid_argument("invalid GPU ordinal");
  }
  std::scoped_lock lock(mutex_);
  const auto found = cache_.find(key(source, destination));
  if (found != cache_.end()) {
    return found->second;
  }
  int accessible = 0;
  DWDP_CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, destination, source));
  if (accessible != 0) {
    int previous = 0;
    DWDP_CUDA_CHECK(cudaGetDevice(&previous));
    DWDP_CUDA_CHECK(cudaSetDevice(destination));
    const auto status = cudaDeviceEnablePeerAccess(source, 0);
    if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
      DWDP_CUDA_CHECK(cudaSetDevice(previous));
      DWDP_CUDA_CHECK(status);
    }
    DWDP_CUDA_CHECK(cudaSetDevice(previous));
  }
  cache_.emplace(key(source, destination), accessible != 0);
  return accessible != 0;
}

} // namespace dwdp::communication
