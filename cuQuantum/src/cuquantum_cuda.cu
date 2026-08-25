#include "cuquantum_api.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Complex = cuDoubleComplex;

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " +
                                 cudaGetErrorString(status));
    }
}

void check_custatevec(custatevecStatus_t status, const char* context) {
    if (status == CUSTATEVEC_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(context) + ": cuStateVec status " +
                             std::to_string(static_cast<int>(status)));
}

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t size) { allocate(size); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t size) {
        if (ptr) check_cuda(cudaFree(ptr), "cudaFree");
        ptr = nullptr; bytes = size;
        if (size) check_cuda(cudaMalloc(&ptr, size), "cudaMalloc");
    }
};

struct Runtime {
    custatevecHandle_t handle = nullptr;
    DeviceBuffer workspace;
    size_t workspace_size = 0;
    bool workspace_ready = false;
    Runtime() { check_custatevec(custatevecCreate(&handle), "custatevecCreate"); }
    Runtime(const Runtime&) = delete;
    ~Runtime() { if (handle) custatevecDestroy(handle); }
};

__host__ __device__ Complex make_complex(double real, double imag) {
    return make_cuDoubleComplex(real, imag);
}

std::vector<Complex> matrix_1q(int kind, double theta, bool derivative) {
    const double c = std::cos(theta / 2.0), s = std::sin(theta / 2.0);
    std::vector<Complex> m(4);
    if (kind == CUQUANTUM_GATE_RX) {
        m = {make_complex(derivative ? -s / 2 : c, 0),
             make_complex(0, derivative ? -c / 2 : -s),
             make_complex(0, derivative ? -c / 2 : -s),
             make_complex(derivative ? -s / 2 : c, 0)};
    } else if (kind == CUQUANTUM_GATE_RY) {
        m = {make_complex(derivative ? -s / 2 : c, 0),
             make_complex(derivative ? -c / 2 : -s, 0),
             make_complex(derivative ? c / 2 : s, 0),
             make_complex(derivative ? -s / 2 : c, 0)};
    } else {
        const Complex e0 = make_complex(std::cos(theta / 2), -std::sin(theta / 2));
        const Complex e1 = make_complex(std::cos(theta / 2), std::sin(theta / 2));
        m = {derivative ? make_complex(-0.5 * std::sin(theta / 2), -0.5 * std::cos(theta / 2)) : e0,
             make_complex(0, 0), make_complex(0, 0),
             derivative ? make_complex(-0.5 * std::sin(theta / 2), 0.5 * std::cos(theta / 2)) : e1};
    }
    return m;
}

void apply_matrix(Runtime& rt, Complex* state, int qubits,
                  const std::vector<Complex>& matrix, const int32_t* targets,
                  uint32_t target_count, const int32_t* controls,
                  uint32_t control_count, bool adjoint) {
    if (!rt.workspace_ready) {
        check_custatevec(custatevecApplyMatrixGetWorkspaceSize(
            rt.handle, CUDA_C_64F, static_cast<uint32_t>(qubits), matrix.data(),
            CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint ? 1 : 0,
            target_count, control_count, CUSTATEVEC_COMPUTE_64F,
            &rt.workspace_size), "custatevecApplyMatrixGetWorkspaceSize");
        rt.workspace.allocate(rt.workspace_size);
        rt.workspace_ready = true;
    }
    check_custatevec(custatevecApplyMatrix(
        rt.handle, state, CUDA_C_64F, static_cast<uint32_t>(qubits),
        matrix.data(), CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
        adjoint ? 1 : 0, targets, target_count, controls, nullptr,
        control_count, CUSTATEVEC_COMPUTE_64F, rt.workspace.ptr,
        rt.workspace_size),
        "custatevecApplyMatrix");
}

void apply_gate(Runtime& rt, Complex* state, int qubits,
                const CuQuantumGate& gate, bool adjoint) {
    if (gate.kind == CUQUANTUM_GATE_RX || gate.kind == CUQUANTUM_GATE_RY ||
        gate.kind == CUQUANTUM_GATE_RZ) {
        custatevecPauli_t pauli = gate.kind == CUQUANTUM_GATE_RX
                                      ? CUSTATEVEC_PAULI_X
                                      : gate.kind == CUQUANTUM_GATE_RY
                                            ? CUSTATEVEC_PAULI_Y
                                            : CUSTATEVEC_PAULI_Z;
        const int32_t target = gate.wire0;
        const double theta = adjoint ? -gate.angle : gate.angle;
        check_custatevec(custatevecApplyPauliRotation(
            rt.handle, state, CUDA_C_64F, static_cast<uint32_t>(qubits),
            -theta / 2.0, &pauli, &target, 1, nullptr, nullptr, 0),
            "custatevecApplyPauliRotation");
        return;
    }
    if (gate.kind == CUQUANTUM_GATE_RZZ) {
        const custatevecPauli_t paulis[2] = {CUSTATEVEC_PAULI_Z, CUSTATEVEC_PAULI_Z};
        const int32_t targets[2] = {gate.wire0, gate.wire1};
        const double theta = adjoint ? -gate.angle : gate.angle;
        check_custatevec(custatevecApplyPauliRotation(
            rt.handle, state, CUDA_C_64F, static_cast<uint32_t>(qubits),
            -theta / 2.0, paulis, targets, 2, nullptr, nullptr, 0),
            "custatevecApplyPauliRotation(RZZ)");
        return;
    }
    const std::vector<Complex> matrix = gate.kind == CUQUANTUM_GATE_CNOT
                                            ? std::vector<Complex>{
                                                  make_complex(0, 0), make_complex(1, 0),
                                                  make_complex(1, 0), make_complex(0, 0)}
                                            : matrix_1q(gate.kind, gate.angle, false);
    const int32_t target = gate.kind == CUQUANTUM_GATE_CNOT ? gate.wire1 : gate.wire0;
    const int32_t control = gate.wire0;
    apply_matrix(rt, state, qubits, matrix, &target, 1,
                 gate.kind == CUQUANTUM_GATE_CNOT ? &control : nullptr,
                 gate.kind == CUQUANTUM_GATE_CNOT ? 1 : 0, adjoint);
}

__global__ void hamiltonian_kernel(const Complex* state, Complex* lambda,
                                    uint64_t count, int qubits, int circuit,
                                    double* energy) {
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        Complex h = make_complex(0, 0);
        if (circuit == 3 || circuit == 8 || circuit == 9 || circuit == 10) {
            int zz = 0;
            for (int q = 0; q < qubits; ++q)
                zz += (((index >> q) & 1) ==
                       ((index >> ((q + 1) % qubits)) & 1)) ? 1 : -1;
            h = cuCmul(state[index], make_complex(0.5 * (zz - qubits), 0));
        } else if (circuit == 5 || circuit == 7) {
            const int target = circuit == 5 ? qubits - 1 : 0;
            h = cuCmul(state[index], make_complex(
                (index & (1ull << target)) ? -1.0 : 1.0, 0));
        } else if (circuit == 6) {
            for (int q = 0; q < qubits; ++q)
                h = cuCadd(h, cuCmul(state[index ^ (1ull << q)],
                                     make_complex(1.0 / qubits, 0)));
        } else if (circuit == 4 || circuit == 11) {
            int zz = 0;
            for (int q = 0; q < qubits; ++q) {
                const int next = (q + 1) % qubits;
                const bool unequal = ((index >> q) & 1) !=
                                     ((index >> next) & 1);
                zz += unequal ? -1 : 1;
                if (unequal)
                    h = cuCadd(h, cuCmul(
                        state[index ^ (1ull << q) ^ (1ull << next)],
                        make_complex(2, 0)));
            }
            h = cuCadd(h, cuCmul(state[index], make_complex(0.5 * zz, 0)));
        } else {
            int zz = 0;
            for (int q = 0; q < qubits; ++q)
                zz += (((index >> q) & 1) ==
                       ((index >> ((q + 1) % qubits)) & 1)) ? 1 : -1;
            h = cuCmul(state[index], make_complex(-static_cast<double>(zz), 0));
            for (int q = 0; q < qubits; ++q)
                h = cuCadd(h, cuCmul(make_complex(-1, 0),
                                     state[index ^ (1ull << q)]));
        }
        lambda[index] = h;
        atomicAdd(energy, cuCreal(cuCmul(cuConj(state[index]), h)));
    }
}

__global__ void pauli_gradient_kernel(const Complex* state,
                                      const Complex* lambda, uint64_t count,
                                      int wire0, int wire1, int pauli_kind,
                                      double* gradient) {
    double local = 0.0;
    const uint64_t mask0 = 1ull << wire0;
    const uint64_t mask1 = wire1 >= 0 ? (1ull << wire1) : 0ull;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        Complex p = state[index];
        if (pauli_kind == 0) { // X
            p = state[index ^ mask0];
        } else if (pauli_kind == 1) { // Y
            const Complex source = state[index ^ mask0];
            p = ((index & mask0) != 0) ? make_complex(-source.y, source.x)
                                      : make_complex(source.y, -source.x);
        } else if (pauli_kind == 2) { // Z
            if (index & mask0) p = make_complex(-p.x, -p.y);
        } else { // ZZ
            if (((index & mask0) != 0) != ((index & mask1) != 0))
                p = make_complex(-p.x, -p.y);
        }
        // 2 Re <lambda | (-i P / 2) | state>.
        const Complex term = cuCmul(cuConj(lambda[index]),
                                    make_complex(p.y, -p.x));
        local += term.x;
    }
    atomicAdd(gradient, local);
}

}  // namespace

extern "C" int cuquantum_energy_and_grad(
    int circuit, int qubits, const CuQuantumGate* gates, size_t gate_count,
    size_t parameter_count, double* out_energy, double* out_gradient,
    char* error_message, size_t error_message_size) {
    try {
        if (!gates || !out_energy || !out_gradient || qubits < 1 || qubits > 30)
            throw std::invalid_argument("invalid cuQuantum arguments");
        Runtime rt;
        const size_t state_count = 1ull << qubits;
        DeviceBuffer current(state_count * sizeof(Complex));
        DeviceBuffer lambda(state_count * sizeof(Complex));
        DeviceBuffer energy_device(sizeof(double));
        DeviceBuffer gradient_device(parameter_count * sizeof(double));
        std::vector<Complex> zero(state_count, make_complex(0, 0));
        zero[0] = make_complex(1, 0);
        check_cuda(cudaMemcpy(current.ptr, zero.data(), zero.size() * sizeof(Complex), cudaMemcpyHostToDevice), "copy zero state");
        for (size_t i = 0; i < gate_count; ++i) apply_gate(rt, static_cast<Complex*>(current.ptr), qubits, gates[i], false);
        check_cuda(cudaMemset(energy_device.ptr, 0, sizeof(double)), "zero energy");
        const int blocks = static_cast<int>(std::min<uint64_t>(
            4096, (state_count + 255) / 256));
        hamiltonian_kernel<<<blocks, 256>>>(
            static_cast<const Complex*>(current.ptr),
            static_cast<Complex*>(lambda.ptr), state_count, qubits, circuit,
            static_cast<double*>(energy_device.ptr));
        check_cuda(cudaGetLastError(), "hamiltonian kernel");
        check_cuda(cudaMemcpy(out_energy, energy_device.ptr, sizeof(double),
                              cudaMemcpyDeviceToHost), "copy energy");
        check_cuda(cudaMemset(gradient_device.ptr, 0,
                              parameter_count * sizeof(double)),
                   "zero gradients");
        for (size_t reverse = gate_count; reverse-- > 0;) {
            const CuQuantumGate& gate = gates[reverse];
            if (gate.parameter >= 0 && static_cast<size_t>(gate.parameter) < parameter_count) {
                const int pauli_kind = gate.kind == CUQUANTUM_GATE_RX
                                            ? 0
                                            : gate.kind == CUQUANTUM_GATE_RY
                                                  ? 1
                                                  : gate.kind == CUQUANTUM_GATE_RZ
                                                        ? 2
                                                        : 3;
                pauli_gradient_kernel<<<blocks, 256>>>(
                    static_cast<const Complex*>(current.ptr),
                    static_cast<const Complex*>(lambda.ptr), state_count,
                    gate.wire0, gate.kind == CUQUANTUM_GATE_RZZ ? gate.wire1 : -1,
                    pauli_kind,
                    static_cast<double*>(gradient_device.ptr) + gate.parameter);
                check_cuda(cudaGetLastError(), "inverse-walk gradient kernel");
            }
            apply_gate(rt, static_cast<Complex*>(current.ptr), qubits, gate, true);
            apply_gate(rt, static_cast<Complex*>(lambda.ptr), qubits, gate, true);
        }
        check_cuda(cudaMemcpy(out_gradient, gradient_device.ptr,
                              parameter_count * sizeof(double),
                              cudaMemcpyDeviceToHost), "copy gradients");
        if (error_message && error_message_size) error_message[0] = '\0';
        return 0;
    } catch (const std::exception& exc) {
        if (error_message && error_message_size) std::snprintf(error_message, error_message_size, "%s", exc.what());
        return 1;
    }
}

extern "C" const char* cuquantum_version(void) { return "0.1.0-custatevec"; }
