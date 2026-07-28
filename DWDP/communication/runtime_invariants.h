#pragma once

namespace dwdp::communication {

inline constexpr const char *kRuntimeInvariants[] = {
    "Executor resolves only resident pointers through CommunicationEngine.",
    "Only CommunicationEngine-owned resources create CUDA work.",
    "Every copy publication has exactly one expert completion event.",
    "Current staging storage is never a copy destination.",
    "An active expert occupies one cache slot.",
    "Pinned experts are never evicted.",
    "Imported IPC mappings are unique per exported expert and reused.",
};

} // namespace dwdp::communication
