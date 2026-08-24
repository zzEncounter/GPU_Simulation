#include "kernels/rotation.cuh"
#include "runtime/lookups.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace sad;

namespace {

struct DeviceMaps {
    int* selected = nullptr;
    int* masks = nullptr;
    int phases = 0;

    DeviceMaps(const std::vector<int>& host_selected,
               const std::vector<int>& host_masks)
        : phases(static_cast<int>(host_masks.size())) {
        if (phases == 0) return;
        SAD_CUDA_CHECK(cudaMalloc(&selected,
                                  host_selected.size() * sizeof(int)));
        SAD_CUDA_CHECK(cudaMalloc(&masks, host_masks.size() * sizeof(int)));
        SAD_CUDA_CHECK(cudaMemcpy(selected,
                                  host_selected.data(),
                                  host_selected.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        SAD_CUDA_CHECK(cudaMemcpy(masks,
                                  host_masks.data(),
                                  host_masks.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
    }

    ~DeviceMaps() {
        cudaFree(selected);
        cudaFree(masks);
    }

    DeviceMaps(const DeviceMaps&) = delete;
    DeviceMaps& operator=(const DeviceMaps&) = delete;
};

void build_subset_phase_maps(int qubits,
                             int first_target,
                             int target_count,
                             std::vector<int>* selected,
                             std::vector<int>* masks) {
    if (target_count == 0) {
        selected->clear();
        masks->clear();
        return;
    }
    const int tile_bits = std::min(qubits, kForwardTileBits);
    const int phases = (target_count + tile_bits - 1) / tile_bits;
    selected->assign(phases * kForwardTileBits, -1);
    masks->assign(phases, 0);
    int target = first_target;
    for (int phase = 0; phase < phases; ++phase) {
        const int base = phase * kForwardTileBits;
        const int count = std::min(tile_bits,
                                   first_target + target_count - target);
        for (int slot = 0; slot < count; ++slot, ++target) {
            (*selected)[base + slot] = target;
            (*masks)[phase] |= 1 << slot;
        }
        for (int slot = count; slot < tile_bits; ++slot) {
            for (int filler = 0; filler < qubits; ++filler) {
                bool used = false;
                for (int other = 0; other < tile_bits; ++other) {
                    used |= (*selected)[base + other] == filler;
                }
                if (!used) {
                    (*selected)[base + slot] = filler;
                    break;
                }
            }
        }
    }
}

void set_access_policy(Complex<double>* base,
                       std::size_t bytes,
                       const cudaDeviceProp& properties,
                       bool persisting) {
    cudaStreamAttrValue attribute{};
    if (persisting) {
        const std::size_t window = std::min<std::size_t>(
            bytes, properties.accessPolicyMaxWindowSize);
        attribute.accessPolicyWindow.base_ptr = base;
        attribute.accessPolicyWindow.num_bytes = window;
        attribute.accessPolicyWindow.hitRatio = std::min(
            1.0,
            static_cast<double>(properties.persistingL2CacheMaxSize) /
                static_cast<double>(window));
        attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    } else {
        attribute.accessPolicyWindow.num_bytes = 0;
    }
    SAD_CUDA_CHECK(cudaStreamSetAttribute(
        nullptr, cudaStreamAttributeAccessPolicyWindow, &attribute));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
                     "usage: %s QUBITS baseline|blocked CHUNK_BITS "
                     "LOW_TARGETS normal|persist ITERATIONS\n",
                     argv[0]);
        return 2;
    }
    const int qubits = std::stoi(argv[1]);
    const std::string mode = argv[2];
    const int chunk_bits = std::stoi(argv[3]);
    const int low_targets = std::stoi(argv[4]);
    const bool persisting = std::string(argv[5]) == "persist";
    const int iterations = std::stoi(argv[6]);
    if ((mode != "baseline" && mode != "blocked") ||
        (std::string(argv[5]) != "normal" && !persisting) ||
        qubits < kForwardTileBits || qubits >= 31 || iterations <= 0) {
        return 2;
    }
    if (mode == "blocked" &&
        (chunk_bits < kForwardTileBits || chunk_bits > qubits ||
         low_targets <= 0 || low_targets > chunk_bits ||
         low_targets > qubits)) {
        return 2;
    }

    cudaDeviceProp properties{};
    SAD_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (persisting) {
        SAD_CUDA_CHECK(cudaDeviceSetLimit(
            cudaLimitPersistingL2CacheSize,
            properties.persistingL2CacheMaxSize));
    }

    std::vector<int> baseline_selected;
    std::vector<int> baseline_masks;
    build_phase_maps(qubits,
                     kForwardTileBits,
                     false,
                     &baseline_selected,
                     &baseline_masks);
    DeviceMaps baseline_maps(baseline_selected, baseline_masks);

    std::vector<int> low_selected;
    std::vector<int> low_masks;
    std::vector<int> high_selected;
    std::vector<int> high_masks;
    if (mode == "blocked") {
        build_subset_phase_maps(chunk_bits,
                                0,
                                low_targets,
                                &low_selected,
                                &low_masks);
        build_subset_phase_maps(qubits,
                                low_targets,
                                qubits - low_targets,
                                &high_selected,
                                &high_masks);
    }
    DeviceMaps low_maps(low_selected, low_masks);
    DeviceMaps high_maps(high_selected, high_masks);

    const std::uint64_t amplitudes = 1ull << qubits;
    const std::size_t state_bytes = amplitudes * sizeof(Complex<double>);
    Complex<double>* state = nullptr;
    RotationCoefficients<double>* coefficients = nullptr;
    SAD_CUDA_CHECK(cudaMalloc(&state, state_bytes));
    SAD_CUDA_CHECK(cudaMalloc(&coefficients,
                              qubits * sizeof(*coefficients)));
    SAD_CUDA_CHECK(cudaMemset(state, 0, state_bytes));
    const Complex<double> one = make_complex<double>(1.0, 0.0);
    SAD_CUDA_CHECK(cudaMemcpy(state,
                              &one,
                              sizeof(one),
                              cudaMemcpyHostToDevice));
    std::vector<RotationCoefficients<double>> host_coefficients(
        qubits, {0.123, 0.99240677});
    SAD_CUDA_CHECK(cudaMemcpy(coefficients,
                              host_coefficients.data(),
                              qubits * sizeof(*coefficients),
                              cudaMemcpyHostToDevice));

    const std::uint64_t chunk_amplitudes =
        mode == "blocked" ? 1ull << chunk_bits : amplitudes;
    const std::size_t chunk_bytes =
        chunk_amplitudes * sizeof(Complex<double>);
    const std::uint64_t chunks = amplitudes / chunk_amplitudes;

    auto launch = [&]() {
        if (mode == "baseline") {
            if (persisting) {
                set_access_policy(state, state_bytes, properties, true);
            }
            launch_non_diagonal_forward<double, NonDiagonalGate::RY>(
                state,
                coefficients,
                qubits,
                0,
                baseline_maps.selected,
                baseline_maps.masks,
                baseline_maps.phases,
                properties.multiProcessorCount);
            return;
        }
        for (std::uint64_t chunk = 0; chunk < chunks; ++chunk) {
            Complex<double>* chunk_state = state + chunk * chunk_amplitudes;
            if (persisting) {
                set_access_policy(chunk_state, chunk_bytes, properties, true);
            }
            launch_non_diagonal_forward<double, NonDiagonalGate::RY>(
                chunk_state,
                coefficients,
                chunk_bits,
                0,
                low_maps.selected,
                low_maps.masks,
                low_maps.phases,
                properties.multiProcessorCount);
        }
        if (high_maps.phases > 0) {
            if (persisting) {
                set_access_policy(state, state_bytes, properties, false);
            }
            launch_non_diagonal_forward<double, NonDiagonalGate::RY>(
                state,
                coefficients,
                qubits,
                0,
                high_maps.selected,
                high_maps.masks,
                high_maps.phases,
                properties.multiProcessorCount);
        }
    };

    double verification_max_error =
        std::numeric_limits<double>::quiet_NaN();
    if (mode == "blocked" && qubits <= 20) {
        Complex<double>* reference = nullptr;
        SAD_CUDA_CHECK(cudaMalloc(&reference, state_bytes));
        SAD_CUDA_CHECK(cudaMemset(reference, 0, state_bytes));
        SAD_CUDA_CHECK(cudaMemcpy(reference,
                                  &one,
                                  sizeof(one),
                                  cudaMemcpyHostToDevice));
        launch_non_diagonal_forward<double, NonDiagonalGate::RY>(
            reference,
            coefficients,
            qubits,
            0,
            baseline_maps.selected,
            baseline_maps.masks,
            baseline_maps.phases,
            properties.multiProcessorCount);
        launch();
        SAD_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<Complex<double>> host_reference(amplitudes);
        std::vector<Complex<double>> host_blocked(amplitudes);
        SAD_CUDA_CHECK(cudaMemcpy(host_reference.data(),
                                  reference,
                                  state_bytes,
                                  cudaMemcpyDeviceToHost));
        SAD_CUDA_CHECK(cudaMemcpy(host_blocked.data(),
                                  state,
                                  state_bytes,
                                  cudaMemcpyDeviceToHost));
        verification_max_error = 0.0;
        for (std::uint64_t index = 0; index < amplitudes; ++index) {
            verification_max_error = std::max(
                verification_max_error,
                std::hypot(host_reference[index].real -
                               host_blocked[index].real,
                           host_reference[index].imag -
                               host_blocked[index].imag));
        }
        cudaFree(reference);
        SAD_CUDA_CHECK(cudaMemset(state, 0, state_bytes));
        SAD_CUDA_CHECK(cudaMemcpy(state,
                                  &one,
                                  sizeof(one),
                                  cudaMemcpyHostToDevice));
    }

    for (int warmup = 0; warmup < 3; ++warmup) launch();
    SAD_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    SAD_CUDA_CHECK(cudaEventCreate(&start));
    SAD_CUDA_CHECK(cudaEventCreate(&stop));
    SAD_CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) launch();
    SAD_CUDA_CHECK(cudaEventRecord(stop));
    SAD_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    SAD_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    const int low_phases = mode == "blocked" ? low_maps.phases : 0;
    const int high_phases = mode == "blocked" ? high_maps.phases : 0;
    std::printf(
        "%s,%s,%d,%d,%d,%llu,%d,%d,%d,%zu,%zu,%.9f,%.17g\n",
        mode.c_str(),
        persisting ? "persist" : "normal",
        qubits,
        chunk_bits,
        low_targets,
        static_cast<unsigned long long>(chunks),
        baseline_maps.phases,
        low_phases,
        high_phases,
        state_bytes,
        chunk_bytes,
        elapsed_ms / iterations,
        verification_max_error);

    if (persisting) {
        set_access_policy(state, state_bytes, properties, false);
        SAD_CUDA_CHECK(cudaCtxResetPersistingL2Cache());
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(state);
    cudaFree(coefficients);
    return 0;
}
