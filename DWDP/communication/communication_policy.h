#pragma once

#include "weight_manager.h"

namespace dwdp::communication {

enum class CommunicationPath : unsigned char { kLocal, kP2P, kIPC, kCopy };

struct CommunicationDecision {
  CommunicationPath path;
};

class CommunicationPolicy final {
 public:
  explicit CommunicationPolicy(int local_gpu) : local_gpu_(local_gpu) {
  }

  [[nodiscard]] CommunicationDecision decide(const ExpertRecord& record, bool peer_access) const;

 private:
  int local_gpu_;
};

}  // namespace dwdp::communication
