#include "topology.h"

#include "cuda_check.h"
#include <stdexcept>
namespace dwdp::communication {
long long PeerTopology::key(int a, int b) { return (static_cast<long long>(a) << 32) | static_cast<unsigned int>(b); }
bool PeerTopology::canAccess(int source, int destination) { if (source < 0 || destination < 0) throw std::invalid_argument("invalid GPU ordinal"); std::scoped_lock lock(mutex_); const auto found = cache_.find(key(source, destination)); if (found != cache_.end()) return found->second; int accessible = 0; DWDP_CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, destination, source)); if (accessible != 0) { int previous = 0; DWDP_CUDA_CHECK(cudaGetDevice(&previous)); DWDP_CUDA_CHECK(cudaSetDevice(destination)); const auto status = cudaDeviceEnablePeerAccess(source, 0); if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) { cudaSetDevice(previous); DWDP_CUDA_CHECK(status); } cudaGetLastError(); DWDP_CUDA_CHECK(cudaSetDevice(previous)); } cache_.emplace(key(source, destination), accessible != 0); return accessible != 0; }
}  // namespace dwdp::communication
