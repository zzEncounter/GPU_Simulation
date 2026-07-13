#include "ising_cuda_backend.hpp"
#include "ising_cuda_backend_internal.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace standalone_backend {
namespace {

using detail::Complex;
template <typename T> using DeviceBuffer = detail::DeviceBuffer<T>;

struct StageProfiler {
    bool enabled;
    std::vector<std::pair<std::string, double>> timings_ms;
    cudaEvent_t start{nullptr};
    cudaEvent_t stop{nullptr};

    explicit StageProfiler(bool enabled_) : enabled(enabled_) {
        if (!enabled) {
            return;
        }
        detail::check_cuda(cudaEventCreate(&start), "cudaEventCreate profiler start");
        detail::check_cuda(cudaEventCreate(&stop), "cudaEventCreate profiler stop");
    }

    StageProfiler(const StageProfiler &) = delete;
    auto operator=(const StageProfiler &) -> StageProfiler & = delete;

    ~StageProfiler() {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
    }
};

template <typename Func>
auto cpu_stage(StageProfiler &profiler, const char *name, Func &&func)
    -> decltype(func()) {
    if (!profiler.enabled) {
        return func();
    }
    const auto start = std::chrono::steady_clock::now();
    if constexpr (std::is_void_v<std::invoke_result_t<Func>>) {
        func();
        const auto stop = std::chrono::steady_clock::now();
        profiler.timings_ms.emplace_back(
            name,
            std::chrono::duration<double, std::milli>(stop - start).count());
    } else {
        auto result = func();
        const auto stop = std::chrono::steady_clock::now();
        profiler.timings_ms.emplace_back(
            name,
            std::chrono::duration<double, std::milli>(stop - start).count());
        return result;
    }
}

template <typename Func>
auto gpu_stage(StageProfiler &profiler, const char *name, Func &&func)
    -> decltype(func()) {
    if (!profiler.enabled) {
        return func();
    }
    detail::check_cuda(cudaEventRecord(profiler.start, 0),
                       "cudaEventRecord profiler start");
    if constexpr (std::is_void_v<std::invoke_result_t<Func>>) {
        func();
        detail::check_cuda(cudaEventRecord(profiler.stop, 0),
                           "cudaEventRecord profiler stop");
        detail::check_cuda(cudaEventSynchronize(profiler.stop),
                           "cudaEventSynchronize profiler stop");
        float elapsed_ms = 0.0F;
        detail::check_cuda(
            cudaEventElapsedTime(&elapsed_ms, profiler.start, profiler.stop),
            "cudaEventElapsedTime profiler");
        profiler.timings_ms.emplace_back(name, static_cast<double>(elapsed_ms));
    } else {
        auto result = func();
        detail::check_cuda(cudaEventRecord(profiler.stop, 0),
                           "cudaEventRecord profiler stop");
        detail::check_cuda(cudaEventSynchronize(profiler.stop),
                           "cudaEventSynchronize profiler stop");
        float elapsed_ms = 0.0F;
        detail::check_cuda(
            cudaEventElapsedTime(&elapsed_ms, profiler.start, profiler.stop),
            "cudaEventElapsedTime profiler");
        profiler.timings_ms.emplace_back(name, static_cast<double>(elapsed_ms));
        return result;
    }
}

enum class GradientStrategy {
    InverseWalk,
    InverseWalkCuQuantum,
    RyrzFused,
    StructuredAdjoint,
    DenseScan
};

constexpr std::size_t STRUCTURED_ROTATION_CHUNK_WIRES = 8;

auto is_structured_adjoint_family(GradientStrategy strategy) -> bool {
    return strategy == GradientStrategy::StructuredAdjoint;
}

auto build_ring_ops(std::size_t num_qubits, std::size_t num_layers,
                    const double *params,
                    bool fuse_ring_cnot_layer) -> std::vector<OpDesc> {
    std::vector<OpDesc> ops;
    ops.reserve(
        num_layers * (num_qubits + (fuse_ring_cnot_layer ? 1 : num_qubits)));

    for (std::size_t layer = 0; layer < num_layers; layer++) {
        for (std::size_t wire = 0; wire < num_qubits; wire++) {
            const auto base = (layer * num_qubits + wire) * 2;
            const double theta0 = params != nullptr ? params[base] : 0.0;
            const double theta1 = params != nullptr ? params[base + 1] : 0.0;
            ops.push_back({OpKind::FusedRYRZ, wire, 0, theta0,
                           theta1, base, base + 1, true});
        }
        if (fuse_ring_cnot_layer) {
            ops.push_back(
                {OpKind::RingCNOTLayer, 0, 0, 0.0, 0.0, 0, 0, false});
        } else {
            for (std::size_t wire = 0; wire < num_qubits; wire++) {
                ops.push_back({OpKind::CNOT, wire, (wire + 1) % num_qubits, 0.0,
                               0.0, 0, 0, false});
            }
        }
    }
    return ops;
}

auto build_pennylane_gate_level_ops(std::size_t num_qubits,
                                    std::size_t num_layers,
                                    const double *params)
    -> std::vector<OpDesc> {
    std::vector<OpDesc> ops;
    ops.reserve(3 * num_layers * num_qubits);

    for (std::size_t layer = 0; layer < num_layers; layer++) {
        for (std::size_t wire = 0; wire < num_qubits; wire++) {
            const auto base = (layer * num_qubits + wire) * 2;
            const double theta0 = params != nullptr ? params[base] : 0.0;
            const double theta1 = params != nullptr ? params[base + 1] : 0.0;
            ops.push_back(
                {OpKind::RY, wire, 0, theta0, 0.0, base, 0, true});
            ops.push_back(
                {OpKind::RZ, wire, 0, theta1, 0.0, base + 1, 0, true});
        }
        for (std::size_t wire = 0; wire < num_qubits; wire++) {
            ops.push_back(
                {OpKind::CNOT, wire, (wire + 1) % num_qubits, 0.0, 0.0, 0, 0,
                 false});
        }
    }
    return ops;
}

auto build_ops_for_strategy(std::size_t num_qubits, std::size_t num_layers,
                            const double *params, GradientStrategy strategy)
    -> std::vector<OpDesc> {
    if (strategy == GradientStrategy::RyrzFused) {
        return build_ring_ops(num_qubits, num_layers, params, false);
    }
    if (strategy == GradientStrategy::DenseScan) {
        return build_ring_ops(num_qubits, num_layers, params, true);
    }
    return build_pennylane_gate_level_ops(num_qubits, num_layers, params);
}

void copy_device_buffer(Complex *dst, const Complex *src, std::size_t size,
                        cudaStream_t stream = 0) {
    detail::check_cuda(cudaMemcpyAsync(dst, src, sizeof(Complex) * size,
                                       cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpyAsyncDeviceToDevice");
}

void apply_op_inplace(Complex *state, std::size_t size, const OpDesc &op,
                      bool inverse, Complex *scratch = nullptr,
                      std::size_t num_qubits = 0) {
    switch (op.kind) {
    case OpKind::RY:
        detail::launch_apply_ry(state, size, op.wire0,
                                inverse ? -op.theta0 : op.theta0);
        break;
    case OpKind::RZ:
        detail::launch_apply_rz(state, size, op.wire0,
                                inverse ? -op.theta0 : op.theta0);
        break;
    case OpKind::CNOT:
        detail::launch_apply_cnot(state, size, op.wire0, op.wire1);
        break;
    case OpKind::FusedRYRZ:
        detail::launch_apply_ryrz(state, size, op.wire0, op.theta0, op.theta1,
                                  inverse);
        break;
    case OpKind::RotationLayer:
        throw std::runtime_error(
            "RotationLayer is only supported by the dense_scan matrix path.");
    case OpKind::RingCNOTLayer:
        if (scratch == nullptr) {
            throw std::runtime_error(
                "RingCNOTLayer requires a scratch buffer for fused execution.");
        }
        detail::launch_apply_ring_cnot_layer(scratch, state, size, num_qubits,
                                             inverse);
        copy_device_buffer(state, scratch, size);
        break;
    }
}

auto validate_and_get_state_size(std::size_t num_qubits,
                                 std::size_t num_layers,
                                 std::size_t num_params) -> std::size_t {
    const auto expected_params = num_qubits * num_layers * 2;
    if (num_params != expected_params) {
        throw std::invalid_argument("num_params does not match the circuit.");
    }
    if (num_qubits >= 63) {
        throw std::invalid_argument(
            "This prototype only supports fewer than 63 qubits.");
    }
    return std::size_t{1} << num_qubits;
}

void validate_num_params(std::size_t expected_params, std::size_t num_params) {
    if (num_params != expected_params) {
        throw std::invalid_argument("num_params does not match the circuit.");
    }
}

auto parse_gradient_strategy(const std::string &strategy)
    -> GradientStrategy {
    if (strategy == "inverse_walk_cuQuantum" ||
        strategy == "inverse_walk_cuquantum") {
        return GradientStrategy::InverseWalkCuQuantum;
    }
    if (strategy == "structured_adjoint") {
        return GradientStrategy::StructuredAdjoint;
    }
    if (strategy == "dense_scan") {
        return GradientStrategy::DenseScan;
    }
    throw std::invalid_argument(
        "Unknown gradient strategy. Expected one of: inverse_walk_cuQuantum, "
        "structured_adjoint, dense_scan.");
}

struct CublasHandle {
    cublasHandle_t handle{nullptr};

    CublasHandle() {
        detail::check_cublas(cublasCreate(&handle), "cublasCreate");
        detail::check_cublas(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
            "cublasSetPointerMode");
    }

    CublasHandle(const CublasHandle &) = delete;
    auto operator=(const CublasHandle &) -> CublasHandle & = delete;

    CublasHandle(CublasHandle &&other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }

    ~CublasHandle() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }
};

struct CustatevecHandle {
    custatevecHandle_t handle{nullptr};

    CustatevecHandle() {
        detail::check_custatevec(custatevecCreate(&handle), "custatevecCreate");
    }

    CustatevecHandle(const CustatevecHandle &) = delete;
    auto operator=(const CustatevecHandle &) -> CustatevecHandle & = delete;

    CustatevecHandle(CustatevecHandle &&other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }

    ~CustatevecHandle() {
        if (handle != nullptr) {
            custatevecDestroy(handle);
        }
    }
};

auto next_power_of_two(std::size_t value) -> std::size_t {
    if (value == 0) {
        return 1;
    }
    std::size_t power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

auto to_int(std::size_t value, const char *context) -> int {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(context) + " does not fit in int.");
    }
    return static_cast<int>(value);
}

#include "ising_cuda_dense_helpers.inc"

} // namespace

struct RingIsingCudaBackend::Impl {
    std::size_t num_qubits;
    std::size_t num_layers;
    double field;
    std::size_t expected_params;
    std::size_t state_size;
    std::size_t num_ops;
    GradientStrategy strategy;
    std::size_t structured_rotation_chunk_width;
    DeviceBuffer<Complex> current;
    DeviceBuffer<Complex> lambda;
    DeviceBuffer<Complex> scratch;
    DeviceBuffer<Complex> cnot_scratch;
    DeviceBuffer<double> gate_level_gradients;
    std::unique_ptr<CustatevecHandle> custatevec;
    DeviceBuffer<std::uint8_t> custatevec_workspace;
    std::size_t custatevec_cnot_workspace_size{0};
    bool custatevec_cnot_workspace_size_cached{false};
    std::unique_ptr<CublasHandle> dense_cublas;
    DeviceBuffer<Complex> dense_gate_mats;
    DeviceBuffer<Complex> dense_hamiltonian;
    DeviceBuffer<int> dense_param_gate_indices;
    DeviceBuffer<Complex> dense_prefix_scan;
    DeviceBuffer<Complex> dense_ones_vector;
    DeviceBuffer<Complex> dense_psi_before;
    DeviceBuffer<Complex> dense_psi_after;
    DeviceBuffer<Complex> dense_forward_states;
    DeviceBuffer<Complex> dense_total_matrix;
    DeviceBuffer<Complex> dense_lambda_k;
    DeviceBuffer<Complex> dense_eta_vector;
    DeviceBuffer<Complex> dense_eta_before;
    DeviceBuffer<Complex> dense_suffix_scan;
    DeviceBuffer<double> dense_gradients;
    DeviceBuffer<double> dense_params;
    DenseGateStorage dense_storage_cache;
    std::vector<std::size_t> dense_param_op_indices;
    std::vector<Complex> dense_hamiltonian_host;
    std::vector<Complex> dense_psi0_host;
    bool dense_static_device_uploaded{false};

    Impl(std::size_t num_qubits_, std::size_t num_layers_, double field_,
         const std::string &gradient_strategy_,
         std::size_t structured_rotation_chunk_width_)
        : num_qubits(num_qubits_), num_layers(num_layers_), field(field_),
          expected_params(num_qubits_ * num_layers_ * 2),
          state_size(validate_and_get_state_size(num_qubits_, num_layers_,
                                                 num_qubits_ * num_layers_ * 2)),
          num_ops(num_layers_ * (num_qubits_ + 1)),
          strategy(parse_gradient_strategy(gradient_strategy_)),
          structured_rotation_chunk_width(structured_rotation_chunk_width_),
          current(state_size), lambda(state_size), scratch(state_size),
          gate_level_gradients(expected_params) {
        if (is_structured_adjoint_family(strategy) &&
            structured_rotation_chunk_width == 0) {
            throw std::invalid_argument(
                "structured_rotation_chunk_width must be at least 1.");
        }
        if (is_structured_adjoint_family(strategy) &&
            structured_rotation_chunk_width > STRUCTURED_ROTATION_CHUNK_WIRES) {
            throw std::invalid_argument(
                "structured_rotation_chunk_width exceeds supported maximum.");
        }
        if (is_structured_adjoint_family(strategy)) {
            cnot_scratch.allocate(state_size);
        }
    }
};

#include "ising_cuda_statevector_modes.inc"

#include "ising_cuda_dense_modes.inc"

RingIsingCudaBackend::RingIsingCudaBackend(std::size_t num_qubits,
                                           std::size_t num_layers,
                                           double field,
                                           const std::string &gradient_strategy,
                                           std::size_t structured_rotation_chunk_width)
    : impl_(std::make_unique<Impl>(num_qubits, num_layers, field,
                                   gradient_strategy,
                                   structured_rotation_chunk_width)) {}

RingIsingCudaBackend::~RingIsingCudaBackend() = default;

RingIsingCudaBackend::RingIsingCudaBackend(RingIsingCudaBackend &&) noexcept =
    default;

auto RingIsingCudaBackend::operator=(RingIsingCudaBackend &&) noexcept
    -> RingIsingCudaBackend & = default;

auto RingIsingCudaBackend::energy_and_grad(const double *params,
                                           std::size_t num_params,
                                           bool compute_gradient,
                                           bool profile)
    -> EnergyGradResult {
    if (!compute_gradient) {
        return run_energy_only(*impl_, params, num_params, profile);
    }
    if (impl_->strategy == GradientStrategy::DenseScan) {
        return run_dense_scan_energy_and_grad_fast(*impl_, params, num_params);
    }
    return run_energy_and_grad_baseline(*impl_, params, num_params,
                                        compute_gradient, profile);
}

auto RingIsingCudaBackend::benchmark_structured_forward_graph(
    const double *params, std::size_t num_params, std::size_t repeats)
    -> CudaGraphBenchmarkResult {
    validate_num_params(impl_->expected_params, num_params);
    if (!is_structured_adjoint_family(impl_->strategy)) {
        throw std::invalid_argument(
            "CUDA graph forward benchmark requires structured_adjoint.");
    }
    if (!structured_uses_product_state_init(*impl_)) {
        throw std::invalid_argument(
            "CUDA graph forward benchmark currently requires product-state "
            "structured initialization.");
    }
    if (repeats == 0) {
        throw std::invalid_argument("repeats must be at least 1.");
    }

    const auto forward_rotation_chunks =
        build_structured_forward_rotation_chunks(*impl_);
    cudaStream_t graph_stream{nullptr};
    detail::check_cuda(
        cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags structured forward graph");

    auto run_forward = [&](cudaStream_t stream) {
        if (!structured_uses_product_state_init(*impl_)) {
            detail::launch_init_zero_state(impl_->current.get(),
                                           impl_->state_size);
        }
        apply_structured_forward_layers(*impl_, params,
                                        forward_rotation_chunks, stream);
    };

    detail::check_cuda(cudaDeviceSynchronize(),
                       "benchmark_structured_forward_graph pre-sync");

    cudaEvent_t start{nullptr};
    cudaEvent_t stop{nullptr};
    detail::check_cuda(cudaEventCreate(&start),
                       "cudaEventCreate graph benchmark start");
    detail::check_cuda(cudaEventCreate(&stop),
                       "cudaEventCreate graph benchmark stop");

    run_forward(graph_stream);
    detail::check_cuda(cudaStreamSynchronize(graph_stream),
                       "benchmark_structured_forward_graph warmup sync");
    auto measure_stream_ms = [&](auto &&body) {
        detail::check_cuda(cudaEventRecord(start, graph_stream),
                           "cudaEventRecord graph benchmark start");
        for (std::size_t repeat = 0; repeat < repeats; repeat++) {
            body();
        }
        detail::check_cuda(cudaEventRecord(stop, graph_stream),
                           "cudaEventRecord graph benchmark stop");
        detail::check_cuda(cudaEventSynchronize(stop),
                           "cudaEventSynchronize graph benchmark stop");
        float elapsed_ms = 0.0F;
        detail::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                           "cudaEventElapsedTime graph benchmark");
        return static_cast<double>(elapsed_ms) /
               static_cast<double>(repeats);
    };
    const double normal_ms =
        measure_stream_ms([&]() { run_forward(graph_stream); });

    cudaGraph_t graph{nullptr};
    cudaGraphExec_t graph_exec{nullptr};
    detail::check_cuda(
        cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeRelaxed),
        "cudaStreamBeginCapture structured forward graph");
    run_forward(graph_stream);
    detail::check_cuda(cudaStreamEndCapture(graph_stream, &graph),
                       "cudaStreamEndCapture structured forward graph");
    detail::check_cuda(cudaGraphInstantiate(&graph_exec, graph, nullptr,
                                            nullptr, 0),
                       "cudaGraphInstantiate structured forward graph");

    detail::check_cuda(cudaGraphLaunch(graph_exec, graph_stream),
                       "cudaGraphLaunch structured forward graph warmup");
    detail::check_cuda(cudaStreamSynchronize(graph_stream),
                       "structured forward graph warmup sync");
    const double graph_ms = measure_stream_ms([&]() {
        detail::check_cuda(cudaGraphLaunch(graph_exec, graph_stream),
                           "cudaGraphLaunch structured forward graph");
    });

    if (graph_exec != nullptr) {
        cudaGraphExecDestroy(graph_exec);
    }
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(graph_stream);

    return {normal_ms, graph_ms};
}

EnergyGradResult energy_and_grad(std::size_t num_qubits,
                                 std::size_t num_layers, double field,
                                 const std::string &gradient_strategy,
                                 const double *params, std::size_t num_params,
                                 bool compute_gradient,
                                 bool profile,
                                 std::size_t structured_rotation_chunk_width) {
    RingIsingCudaBackend backend(num_qubits, num_layers, field,
                                 gradient_strategy,
                                 structured_rotation_chunk_width);
    return backend.energy_and_grad(params, num_params, compute_gradient, profile);
}

} // namespace standalone_backend
