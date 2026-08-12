#include "kernels/xxz.cuh"
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

struct IntegratedMaps {
    std::vector<int> selected;
    std::vector<int> offsets{0};
    std::vector<int> slot_a;
    std::vector<int> slot_b;
    std::vector<int> edges;
};

void fill_selected_phase(int qubits,
                         int tile_bits,
                         const std::vector<int>& phase_qubits,
                         std::vector<int>* selected) {
    const int base = static_cast<int>(selected->size());
    selected->resize(base + tile_bits, -1);
    int filled = 0;
    for (const int qubit : phase_qubits) {
        (*selected)[base + filled++] = qubit;
    }
    for (int qubit = 0; filled < tile_bits && qubit < qubits; ++qubit) {
        if (std::find(phase_qubits.begin(), phase_qubits.end(), qubit) ==
            phase_qubits.end()) {
            (*selected)[base + filled++] = qubit;
        }
    }
}

auto build_integrated_maps(int qubits, int tile_bits) -> IntegratedMaps {
    IntegratedMaps maps;
    tile_bits = std::min(qubits, tile_bits);
    const int block_width = tile_bits & ~1;
    std::vector<std::pair<int, int>> blocks;
    for (int first = 0; first < qubits; first += block_width) {
        const int width = std::min(block_width, qubits - first);
        blocks.emplace_back(first, width);
        std::vector<int> phase_qubits;
        for (int offset = 0; offset < width; ++offset) {
            phase_qubits.push_back(first + offset);
        }
        fill_selected_phase(qubits, tile_bits, phase_qubits, &maps.selected);
        for (int offset = 0; offset + 1 < width; offset += 2) {
            maps.slot_a.push_back(offset);
            maps.slot_b.push_back(offset + 1);
            maps.edges.push_back(first + offset);
        }
        for (int offset = 1; offset + 1 < width; offset += 2) {
            maps.slot_a.push_back(offset);
            maps.slot_b.push_back(offset + 1);
            maps.edges.push_back(first + offset);
        }
        maps.offsets.push_back(static_cast<int>(maps.edges.size()));
    }

    std::vector<int> boundary_qubits;
    std::vector<std::pair<int, int>> boundary_bonds;
    for (std::size_t block = 0; block < blocks.size(); ++block) {
        const int left = blocks[block].first + blocks[block].second - 1;
        const int right = blocks[(block + 1) % blocks.size()].first;
        boundary_bonds.emplace_back(left, right);
        boundary_qubits.push_back(left);
        boundary_qubits.push_back(right);
    }
    fill_selected_phase(qubits, tile_bits, boundary_qubits, &maps.selected);
    for (const auto [left, right] : boundary_bonds) {
        const auto left_it =
            std::find(boundary_qubits.begin(), boundary_qubits.end(), left);
        const auto right_it =
            std::find(boundary_qubits.begin(), boundary_qubits.end(), right);
        maps.slot_a.push_back(
            static_cast<int>(left_it - boundary_qubits.begin()));
        maps.slot_b.push_back(
            static_cast<int>(right_it - boundary_qubits.begin()));
        maps.edges.push_back(left);
    }
    maps.offsets.push_back(static_cast<int>(maps.edges.size()));
    return maps;
}

template <typename T>
__global__ void xxz_integrated_forward_kernel(
    Complex<T>* state,
    const RotationCoefficients<T>* coefficients,
    int qubits,
    int x_parameter_offset,
    int y_parameter_offset,
    int z_parameter_offset,
    const int* selected_maps,
    const int* bond_offsets,
    const int* slot_a,
    const int* slot_b,
    const int* edges,
    int phase_count) {
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* mailbox = reinterpret_cast<Complex<T>*>(dynamic_shared);
    __shared__ std::uint64_t tile_base;
    const int tile_bits = min(qubits, kForwardTileBits);
    const std::uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int phase = 0; phase < phase_count; ++phase) {
        const int* selected = selected_maps + phase * kForwardTileBits;
        for (std::uint64_t tile = blockIdx.x; tile < tile_count;
             tile += gridDim.x) {
            if (tid == 0) {
                tile_base = scatter_tile_assignment<kForwardTileBits>(
                    tile, qubits, selected, tile_bits);
            }
            __syncthreads();
            Complex<T> values[kForwardRegisterAmplitudes];
#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                const std::uint32_t local = static_cast<std::uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kForwardRegisterBits)));
                const std::uint64_t index =
                    tile_base |
                    scatter_local_assignment<kForwardTileBits>(
                        local, selected, tile_bits);
                values[reg] = state[index];
            }
            for (int bond = bond_offsets[phase];
                 bond < bond_offsets[phase + 1];
                 ++bond) {
#pragma unroll
                for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                    const std::uint32_t local = static_cast<std::uint32_t>(
                        lane | (reg << kLaneBits) |
                        (warp << (kLaneBits + kForwardRegisterBits)));
                    mailbox[local] = values[reg];
                }
                __syncthreads();
                const int first_slot = slot_a[bond];
                const int second_slot = slot_b[bond];
                const std::uint32_t pair_mask =
                    (1u << first_slot) | (1u << second_slot);
                const int edge = edges[bond];
                const auto x = coefficients[x_parameter_offset + edge];
                const auto y = coefficients[y_parameter_offset + edge];
                const auto z = coefficients[z_parameter_offset + edge];
#pragma unroll
                for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                    const std::uint32_t local = static_cast<std::uint32_t>(
                        lane | (reg << kLaneBits) |
                        (warp << (kLaneBits + kForwardRegisterBits)));
                    const int z_eigenvalue =
                        ((local >> first_slot) & 1u) ==
                                ((local >> second_slot) & 1u)
                            ? 1
                            : -1;
                    values[reg] = apply_xxz_bond(
                        values[reg],
                        mailbox[local ^ pair_mask],
                        z_eigenvalue,
                        x,
                        y,
                        z);
                }
                __syncthreads();
            }
#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                const std::uint32_t local = static_cast<std::uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kForwardRegisterBits)));
                const std::uint64_t index =
                    tile_base |
                    scatter_local_assignment<kForwardTileBits>(
                        local, selected, tile_bits);
                state[index] = values[reg];
            }
            __syncthreads();
        }
    }
}

template <typename T>
void launch_integrated(Complex<T>* state,
                       const RotationCoefficients<T>* coefficients,
                       int qubits,
                       const int* selected,
                       const int* offsets,
                       const int* slot_a,
                       const int* slot_b,
                       const int* edges,
                       int phase_count,
                       int multiprocessors) {
    const int tile_bits = std::min(qubits, kForwardTileBits);
    const std::uint64_t tile_count = 1ull << (qubits - tile_bits);
    constexpr std::size_t shared_bytes =
        kForwardTileAmplitudes * sizeof(Complex<T>);
    const auto kernel = xxz_integrated_forward_kernel<T>;
    SAD_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    int x_parameter_offset = 0;
    int y_parameter_offset = qubits;
    int z_parameter_offset = 2 * qubits;
    for (int phase = 0; phase < phase_count; ++phase) {
        kernel<<<static_cast<int>(tile_count),
                 kForwardBlockThreads,
                 shared_bytes>>>(state,
                                 coefficients,
                                 qubits,
                                 x_parameter_offset,
                                 y_parameter_offset,
                                 z_parameter_offset,
                                 selected + phase * kForwardTileBits,
                                 offsets + phase,
                                 slot_a,
                                 slot_b,
                                 edges,
                                 1);
    }
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void xxz_integrated_backward_kernel(
    Complex<T>* phi_state,
    Complex<T>* lambda_state,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    int x_parameter_offset,
    int y_parameter_offset,
    int z_parameter_offset,
    const int* selected,
    int bond_begin,
    int bond_end,
    const int* slot_a,
    const int* slot_b,
    const int* edges) {
    __shared__ Complex<T> mailbox[kTileAmplitudes];
    __shared__ double reduction[kBlockThreads];
    __shared__ uint64_t tile_base;
    const int tile_bits = min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (uint64_t tile = blockIdx.x; tile < tile_count;
         tile += gridDim.x) {
        if (tid == 0) {
            tile_base = scatter_tile_assignment<kTileBits>(
                tile, qubits, selected, tile_bits);
        }
        __syncthreads();
        Complex<T> phi[kRegisterAmplitudes];
        Complex<T> lambda[kRegisterAmplitudes];
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const uint32_t local = static_cast<uint32_t>(
                lane | (reg << kLaneBits) |
                (warp << (kLaneBits + kRegisterBits)));
            const bool active = local < (1u << tile_bits);
            const uint64_t index =
                tile_base | scatter_local_assignment<kTileBits>(
                                local, selected, tile_bits);
            phi[reg] = active ? phi_state[index] : make_complex<T>(0, 0);
            lambda[reg] =
                active ? lambda_state[index] : make_complex<T>(0, 0);
        }
        for (int bond = bond_end - 1; bond >= bond_begin; --bond) {
            const int first_slot = slot_a[bond];
            const int second_slot = slot_b[bond];
            const uint32_t pair_mask =
                (1u << first_slot) | (1u << second_slot);
            const int edge = edges[bond];
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                mailbox[local] = phi[reg];
            }
            __syncthreads();
            double x_overlap = 0.0;
            double y_overlap = 0.0;
            double z_overlap = 0.0;
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const int z_eigenvalue =
                        ((local >> first_slot) & 1u) ==
                                ((local >> second_slot) & 1u)
                            ? 1
                            : -1;
                    const Complex<T> partner = mailbox[local ^ pair_mask];
                    x_overlap +=
                        imag_conjugate_product(lambda[reg], partner);
                    y_overlap += imag_conjugate_product(
                        lambda[reg],
                        scale(partner, static_cast<T>(-z_eigenvalue)));
                    z_overlap += imag_conjugate_product(
                        lambda[reg],
                        scale(phi[reg], static_cast<T>(z_eigenvalue)));
                }
            }
            block_atomic_sum(x_overlap,
                             reduction,
                             gradients + x_parameter_offset + edge);
            block_atomic_sum(y_overlap,
                             reduction,
                             gradients + y_parameter_offset + edge);
            block_atomic_sum(z_overlap,
                             reduction,
                             gradients + z_parameter_offset + edge);
            RotationCoefficients<T> x = coefficients[x_parameter_offset + edge];
            RotationCoefficients<T> y = coefficients[y_parameter_offset + edge];
            RotationCoefficients<T> z = coefficients[z_parameter_offset + edge];
            x.sine = -x.sine;
            y.sine = -y.sine;
            z.sine = -z.sine;
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const int z_eigenvalue =
                        ((local >> first_slot) & 1u) ==
                                ((local >> second_slot) & 1u)
                            ? 1
                            : -1;
                    phi[reg] = apply_xxz_bond(phi[reg],
                                               mailbox[local ^ pair_mask],
                                               z_eigenvalue,
                                               x,
                                               y,
                                               z);
                }
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                mailbox[local] = lambda[reg];
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const int z_eigenvalue =
                        ((local >> first_slot) & 1u) ==
                                ((local >> second_slot) & 1u)
                            ? 1
                            : -1;
                    lambda[reg] = apply_xxz_bond(lambda[reg],
                                                  mailbox[local ^ pair_mask],
                                                  z_eigenvalue,
                                                  x,
                                                  y,
                                                  z);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const uint32_t local = static_cast<uint32_t>(
                lane | (reg << kLaneBits) |
                (warp << (kLaneBits + kRegisterBits)));
            if (local < (1u << tile_bits)) {
                const uint64_t index =
                    tile_base | scatter_local_assignment<kTileBits>(
                                    local, selected, tile_bits);
                phi_state[index] = phi[reg];
                lambda_state[index] = lambda[reg];
            }
        }
        __syncthreads();
    }
}

template <typename T>
void launch_integrated_backward(
    Complex<T>* phi,
    Complex<T>* lambda,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    const int* selected,
    const std::vector<int>& host_offsets,
    const int* slot_a,
    const int* slot_b,
    const int* edges) {
    const int tile_bits = std::min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int phase_count = static_cast<int>(host_offsets.size()) - 1;
    for (int step = 0; step < phase_count; ++step) {
        const int phase = phase_count - 1 - step;
        xxz_integrated_backward_kernel<T>
            <<<static_cast<int>(tile_count), kBlockThreads>>>(
                phi,
                lambda,
                coefficients,
                gradients,
                qubits,
                0,
                qubits,
                2 * qubits,
                selected + phase * kTileBits,
                host_offsets[phase],
                host_offsets[phase + 1],
                slot_a,
                slot_b,
                edges);
    }
    SAD_CUDA_CHECK(cudaGetLastError());
}

struct DeviceIntArray {
    int* data = nullptr;
    explicit DeviceIntArray(const std::vector<int>& source) {
        SAD_CUDA_CHECK(cudaMalloc(&data, source.size() * sizeof(int)));
        SAD_CUDA_CHECK(cudaMemcpy(data,
                                  source.data(),
                                  source.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
    }
    ~DeviceIntArray() { cudaFree(data); }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "usage: %s QUBITS forward|backward "
                     "baseline|integrated ITERATIONS\n",
                     argv[0]);
        return 2;
    }
    const int qubits = std::stoi(argv[1]);
    const std::string direction = argv[2];
    const std::string mode = argv[3];
    const int iterations = std::stoi(argv[4]);
    if (qubits < 4 || (qubits & 1) ||
        (direction != "forward" && direction != "backward") ||
        (mode != "baseline" && mode != "integrated") || iterations <= 0) {
        return 2;
    }

    const int tile_bits = direction == "forward" ? kForwardTileBits
                                                    : kTileBits;

    std::vector<int> even_selected;
    std::vector<int> even_counts;
    std::vector<int> odd_selected;
    std::vector<int> odd_counts;
    build_bond_phase_maps(qubits,
                          tile_bits,
                          0,
                          &even_selected,
                          &even_counts);
    build_bond_phase_maps(qubits,
                          tile_bits,
                          1,
                          &odd_selected,
                          &odd_counts);
    const auto integrated = build_integrated_maps(qubits, tile_bits);
    DeviceIntArray d_even_selected(even_selected);
    DeviceIntArray d_even_counts(even_counts);
    DeviceIntArray d_odd_selected(odd_selected);
    DeviceIntArray d_odd_counts(odd_counts);
    DeviceIntArray d_selected(integrated.selected);
    DeviceIntArray d_offsets(integrated.offsets);
    DeviceIntArray d_slot_a(integrated.slot_a);
    DeviceIntArray d_slot_b(integrated.slot_b);
    DeviceIntArray d_edges(integrated.edges);

    cudaDeviceProp properties{};
    SAD_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    const std::uint64_t amplitudes = 1ull << qubits;
    const std::size_t state_bytes = amplitudes * sizeof(Complex<double>);
    Complex<double>* state = nullptr;
    Complex<double>* lambda = nullptr;
    RotationCoefficients<double>* coefficients = nullptr;
    double* gradients = nullptr;
    SAD_CUDA_CHECK(cudaMalloc(&state, state_bytes));
    SAD_CUDA_CHECK(cudaMalloc(&lambda, state_bytes));
    SAD_CUDA_CHECK(cudaMalloc(&coefficients,
                              3 * qubits * sizeof(*coefficients)));
    SAD_CUDA_CHECK(cudaMalloc(&gradients, 3 * qubits * sizeof(*gradients)));
    std::vector<Complex<double>> host_state(amplitudes);
    std::vector<Complex<double>> host_lambda(amplitudes);
    for (std::uint64_t index = 0; index < amplitudes; ++index) {
        const double x = static_cast<double>((index * 17 + 3) & 1023) / 1024.0;
        const double y = static_cast<double>((index * 29 + 5) & 1023) / 1024.0;
        host_state[index] = {x, y};
        host_lambda[index] = {y * 0.75, x * -0.5};
    }
    auto reset = [&](Complex<double>* target_state,
                     Complex<double>* target_lambda,
                     double* target_gradients) {
        SAD_CUDA_CHECK(cudaMemcpy(target_state,
                                  host_state.data(),
                                  state_bytes,
                                  cudaMemcpyHostToDevice));
        SAD_CUDA_CHECK(cudaMemcpy(target_lambda,
                                  host_lambda.data(),
                                  state_bytes,
                                  cudaMemcpyHostToDevice));
        SAD_CUDA_CHECK(cudaMemset(target_gradients,
                                  0,
                                  3 * qubits * sizeof(*target_gradients)));
    };
    reset(state, lambda, gradients);
    std::vector<RotationCoefficients<double>> host_coefficients(3 * qubits);
    for (int parameter = 0; parameter < 3 * qubits; ++parameter) {
        const double half_angle = 0.007 * static_cast<double>(parameter + 1);
        host_coefficients[parameter] = {
            std::sin(half_angle), std::cos(half_angle)};
    }
    SAD_CUDA_CHECK(cudaMemcpy(coefficients,
                              host_coefficients.data(),
                              host_coefficients.size() * sizeof(*coefficients),
                              cudaMemcpyHostToDevice));

    auto launch_baseline = [&](Complex<double>* target) {
        launch_xxz_matching_forward(target,
                                    coefficients,
                                    qubits,
                                    0,
                                    qubits,
                                    2 * qubits,
                                    d_even_selected.data,
                                    d_even_counts.data,
                                    static_cast<int>(even_counts.size()),
                                    properties.multiProcessorCount);
        launch_xxz_matching_forward(target,
                                    coefficients,
                                    qubits,
                                    0,
                                    qubits,
                                    2 * qubits,
                                    d_odd_selected.data,
                                    d_odd_counts.data,
                                    static_cast<int>(odd_counts.size()),
                                    properties.multiProcessorCount);
    };
    auto launch_integrated_schedule = [&](Complex<double>* target) {
        launch_integrated(target,
                          coefficients,
                          qubits,
                          d_selected.data,
                          d_offsets.data,
                          d_slot_a.data,
                          d_slot_b.data,
                          d_edges.data,
                          static_cast<int>(integrated.offsets.size() - 1),
                          properties.multiProcessorCount);
    };
    auto launch_baseline_backward = [&](Complex<double>* target,
                                        Complex<double>* target_lambda,
                                        double* target_gradients) {
        launch_xxz_matching_backward(target,
                                     target_lambda,
                                     coefficients,
                                     target_gradients,
                                     qubits,
                                     0,
                                     qubits,
                                     2 * qubits,
                                     d_odd_selected.data,
                                     d_odd_counts.data,
                                     static_cast<int>(odd_counts.size()),
                                     properties.multiProcessorCount);
        launch_xxz_matching_backward(target,
                                     target_lambda,
                                     coefficients,
                                     target_gradients,
                                     qubits,
                                     0,
                                     qubits,
                                     2 * qubits,
                                     d_even_selected.data,
                                     d_even_counts.data,
                                     static_cast<int>(even_counts.size()),
                                     properties.multiProcessorCount);
    };
    auto launch_integrated_backward_schedule = [&](
                                                   Complex<double>* target,
                                                   Complex<double>* target_lambda,
                                                   double* target_gradients) {
        launch_integrated_backward(target,
                                   target_lambda,
                                   coefficients,
                                   target_gradients,
                                   qubits,
                                   d_selected.data,
                                   integrated.offsets,
                                   d_slot_a.data,
                                   d_slot_b.data,
                                   d_edges.data);
    };
    auto launch = [&](Complex<double>* target,
                      Complex<double>* target_lambda,
                      double* target_gradients) {
        if (direction == "forward") {
            if (mode == "baseline") {
                launch_baseline(target);
            } else {
                launch_integrated_schedule(target);
            }
        } else if (mode == "baseline") {
            launch_baseline_backward(target, target_lambda, target_gradients);
        } else {
            launch_integrated_backward_schedule(
                target, target_lambda, target_gradients);
        }
    };

    double max_error = std::numeric_limits<double>::quiet_NaN();
    double gradient_max_error = std::numeric_limits<double>::quiet_NaN();
    if (mode == "integrated" && qubits <= 20) {
        Complex<double>* reference = nullptr;
        Complex<double>* reference_lambda = nullptr;
        double* reference_gradients = nullptr;
        SAD_CUDA_CHECK(cudaMalloc(&reference, state_bytes));
        SAD_CUDA_CHECK(cudaMalloc(&reference_lambda, state_bytes));
        SAD_CUDA_CHECK(cudaMalloc(&reference_gradients,
                                  3 * qubits * sizeof(*reference_gradients)));
        reset(reference, reference_lambda, reference_gradients);
        reset(state, lambda, gradients);
        if (direction == "forward") {
            launch_baseline(reference);
            launch_integrated_schedule(state);
        } else {
            launch_baseline_backward(
                reference, reference_lambda, reference_gradients);
            launch_integrated_backward_schedule(state, lambda, gradients);
        }
        SAD_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<Complex<double>> host_reference(amplitudes);
        std::vector<Complex<double>> host_integrated(amplitudes);
        SAD_CUDA_CHECK(cudaMemcpy(host_reference.data(),
                                  reference,
                                  state_bytes,
                                  cudaMemcpyDeviceToHost));
        SAD_CUDA_CHECK(cudaMemcpy(host_integrated.data(),
                                  state,
                                  state_bytes,
                                  cudaMemcpyDeviceToHost));
        max_error = 0.0;
        for (std::uint64_t index = 0; index < amplitudes; ++index) {
            max_error = std::max(
                max_error,
                std::hypot(host_reference[index].real -
                               host_integrated[index].real,
                           host_reference[index].imag -
                           host_integrated[index].imag));
        }
        if (direction == "backward") {
            SAD_CUDA_CHECK(cudaMemcpy(host_reference.data(),
                                      reference_lambda,
                                      state_bytes,
                                      cudaMemcpyDeviceToHost));
            SAD_CUDA_CHECK(cudaMemcpy(host_integrated.data(),
                                      lambda,
                                      state_bytes,
                                      cudaMemcpyDeviceToHost));
            for (std::uint64_t index = 0; index < amplitudes; ++index) {
                max_error = std::max(
                    max_error,
                    std::hypot(host_reference[index].real -
                                   host_integrated[index].real,
                               host_reference[index].imag -
                                   host_integrated[index].imag));
            }
            std::vector<double> host_reference_gradients(3 * qubits);
            std::vector<double> host_integrated_gradients(3 * qubits);
            SAD_CUDA_CHECK(cudaMemcpy(host_reference_gradients.data(),
                                      reference_gradients,
                                      3 * qubits * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            SAD_CUDA_CHECK(cudaMemcpy(host_integrated_gradients.data(),
                                      gradients,
                                      3 * qubits * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            gradient_max_error = 0.0;
            for (int parameter = 0; parameter < 3 * qubits; ++parameter) {
                gradient_max_error = std::max(
                    gradient_max_error,
                    std::abs(host_reference_gradients[parameter] -
                             host_integrated_gradients[parameter]));
                if (qubits <= 20 &&
                    std::abs(host_reference_gradients[parameter] -
                             host_integrated_gradients[parameter]) > 1e-6) {
                    std::fprintf(stderr,
                                 "gradient[%d] reference=%.17g "
                                 "integrated=%.17g\n",
                                 parameter,
                                 host_reference_gradients[parameter],
                                 host_integrated_gradients[parameter]);
                }
            }
        }
        cudaFree(reference);
        cudaFree(reference_lambda);
        cudaFree(reference_gradients);
        reset(state, lambda, gradients);
    }

    for (int warmup = 0; warmup < 3; ++warmup) {
        launch(state, lambda, gradients);
    }
    SAD_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    SAD_CUDA_CHECK(cudaEventCreate(&start));
    SAD_CUDA_CHECK(cudaEventCreate(&stop));
    SAD_CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        launch(state, lambda, gradients);
    }
    SAD_CUDA_CHECK(cudaEventRecord(stop));
    SAD_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    SAD_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::printf("%s,%s,%d,%d,%zu,%zu,%.9f,%.17g,%.17g\n",
                direction.c_str(),
                mode.c_str(),
                qubits,
                kForwardTileBits,
                even_counts.size() + odd_counts.size(),
                integrated.offsets.size() - 1,
                elapsed_ms / iterations,
                max_error,
                gradient_max_error);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(state);
    cudaFree(lambda);
    cudaFree(coefficients);
    cudaFree(gradients);
    return 0;
}
