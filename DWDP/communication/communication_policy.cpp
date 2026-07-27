#include "communication_policy.h"

namespace dwdp::communication {
CommunicationDecision CommunicationPolicy::decide(const ExpertRecord& r, bool peer_access) const {
  if (r.state == ResidentState::kActive || r.state == ResidentState::kStaged) return {CommunicationPath::kLocal};
  if (r.has_ipc_handle) return {CommunicationPath::kIPC};
  if (r.owner_gpu != local_gpu_ && peer_access) return {CommunicationPath::kP2P};
  return {CommunicationPath::kCopy};
}
}  // namespace dwdp::communication
