#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include "communication_engine.h"
#include "ipc.h"

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
}
