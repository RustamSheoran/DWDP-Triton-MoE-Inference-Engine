#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include "communication_engine.h"
#include "ipc.h"
#include "topology.h"
#include "transfer_scheduler.h"
#include "communication_policy.h"

namespace py = pybind11;

// Helper to convert python bytes to cudaIpcMemHandle_t
cudaIpcMemHandle_t bytes_to_handle(const py::bytes& b) {
    std::string s = b;
    if (s.size() != sizeof(cudaIpcMemHandle_t)) {
        throw std::invalid_argument("Invalid CUDA IPC handle size");
    }
    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, s.data(), sizeof(cudaIpcMemHandle_t));
    return handle;
}

// Helper to export pointer to bytes
py::bytes export_ipc_handle(intptr_t device_pointer) {
    cudaIpcMemHandle_t handle = dwdp::communication::ipc::exportHandle(reinterpret_cast<const void*>(device_pointer));
    return py::bytes(reinterpret_cast<const char*>(&handle), sizeof(cudaIpcMemHandle_t));
}

PYBIND11_MODULE(dwdp_communication_ext, m) {
    m.doc() = "DWDP Communication Engine C++ Bindings";

    m.attr("NATIVE_AVAILABLE") = true;

    m.def("export_ipc_handle", &export_ipc_handle, "Export a device pointer to a CUDA IPC handle as bytes");

    py::class_<dwdp::communication::CommunicationEngine>(m, "CommunicationEngine")
        .def(py::init<int>(), py::arg("device_id") = 0)
        .def("initialize", &dwdp::communication::CommunicationEngine::initialize)
        .def("shutdown", &dwdp::communication::CommunicationEngine::shutdown)
        .def("register_expert", [](dwdp::communication::CommunicationEngine& self, int expert_id, intptr_t ptr, std::size_t size) {
            self.registerExpert(expert_id, reinterpret_cast<void*>(ptr), size);
        })
        .def("register_ipc_expert", [](dwdp::communication::CommunicationEngine& self, int expert_id, const py::bytes& handle_bytes, std::size_t size) {
            self.registerIPCExpert(expert_id, bytes_to_handle(handle_bytes), size);
        })
        .def("prefetch", &dwdp::communication::CommunicationEngine::prefetch)
        .def("wait", &dwdp::communication::CommunicationEngine::wait)
        .def("get_weight", [](const dwdp::communication::CommunicationEngine& self, int expert_id) {
            return reinterpret_cast<intptr_t>(self.getWeight(expert_id));
        })
        .def("is_resident", &dwdp::communication::CommunicationEngine::isResident)
        .def("get_resident_pointer", [](dwdp::communication::CommunicationEngine& self, int expert_id) {
            return reinterpret_cast<intptr_t>(self.getResidentPointer(expert_id));
        })
        .def("swap_buffers", &dwdp::communication::CommunicationEngine::swapBuffers)
        .def("release", &dwdp::communication::CommunicationEngine::release)
        .def("initialized", &dwdp::communication::CommunicationEngine::initialized);

    py::class_<dwdp::communication::PeerTopology>(m, "PeerTopology")
        .def(py::init<>())
        .def("canAccess", &dwdp::communication::PeerTopology::canAccess, py::arg("source_gpu"), py::arg("destination_gpu"));

    py::enum_<dwdp::communication::TransferState>(m, "TransferState")
        .value("kCreated", dwdp::communication::TransferState::kCreated)
        .value("kQueued", dwdp::communication::TransferState::kQueued)
        .value("kRunning", dwdp::communication::TransferState::kRunning)
        .value("kWaiting", dwdp::communication::TransferState::kWaiting)
        .value("kCompleted", dwdp::communication::TransferState::kCompleted)
        .value("kFailed", dwdp::communication::TransferState::kFailed)
        .value("kCancelled", dwdp::communication::TransferState::kCancelled)
        .export_values();

    py::class_<dwdp::communication::TransferTask, std::shared_ptr<dwdp::communication::TransferTask>>(m, "TransferTask")
        .def(py::init<>())
        .def_readwrite("expert_id", &dwdp::communication::TransferTask::expert_id)
        .def_readwrite("priority", &dwdp::communication::TransferTask::priority)
        .def_readwrite("sequence", &dwdp::communication::TransferTask::sequence)
        .def_readwrite("state", &dwdp::communication::TransferTask::state)
        .def_readwrite("retries", &dwdp::communication::TransferTask::retries);

    py::class_<dwdp::communication::TransferScheduler>(m, "TransferScheduler")
        .def(py::init<>())
        .def("submit", &dwdp::communication::TransferScheduler::submit,
             py::arg("expert_id"), py::arg("priority"), py::arg("completion") = std::function<void(dwdp::communication::TransferState)>())
        .def("take", &dwdp::communication::TransferScheduler::take)
        .def("complete", &dwdp::communication::TransferScheduler::complete, py::arg("task"))
        .def("fail", &dwdp::communication::TransferScheduler::fail, py::arg("task"), py::arg("retryable"))
        .def("cancel", &dwdp::communication::TransferScheduler::cancel, py::arg("expert_id"))
        .def("close", &dwdp::communication::TransferScheduler::close);

    py::enum_<dwdp::communication::CommunicationPath>(m, "CommunicationPath")
        .value("kLocal", dwdp::communication::CommunicationPath::kLocal)
        .value("kP2P", dwdp::communication::CommunicationPath::kP2P)
        .value("kIPC", dwdp::communication::CommunicationPath::kIPC)
        .value("kCopy", dwdp::communication::CommunicationPath::kCopy)
        .export_values();

    py::class_<dwdp::communication::CommunicationDecision>(m, "CommunicationDecision")
        .def(py::init<>())
        .def_readwrite("path", &dwdp::communication::CommunicationDecision::path);

    py::class_<dwdp::communication::ExpertRecord>(m, "ExpertRecord")
        .def(py::init<>());

    py::class_<dwdp::communication::CommunicationPolicy>(m, "CommunicationPolicy")
        .def(py::init<int>(), py::arg("local_gpu"))
        .def("decide", &dwdp::communication::CommunicationPolicy::decide, py::arg("record"), py::arg("peer_access"));
}
