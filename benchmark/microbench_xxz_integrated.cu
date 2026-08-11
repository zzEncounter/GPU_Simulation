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
    selected->resize(base + kForwardTileBits, -1);
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

auto build_integrated_maps(int qubits) -> IntegratedMaps {
    IntegratedMaps maps;
    const int tile_bits = std::min(qubits, kForwardTileBits);
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
    cg::grid_group grid = cg::this_grid();
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
        if (phase + 1 < phase_count) grid.sync();
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
    const auto kernel = xxz_integrated_forward_kernel<T>;
    constexpr std::size_t shared_bytes =
        kForwardTileAmplitudes * sizeof(Complex<T>);
    SAD_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    int active_blocks = 0;
    SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, kForwardBlockThreads, shared_bytes));
    const int grid_size = static_cast<int>(std::min<std::uint64_t>(
        tile_count,
        static_cast<std::uint64_t>(active_blocks) * multiprocessors));
    int x_parameter_offset = 0;
    int y_parameter_offset = qubits;
    int z_parameter_offset = 2 * qubits;
    void* arguments[] = {
        &state,
        const_cast<RotationCoefficients<T>**>(&coefficients),
        &qubits,
        &x_parameter_offset,
        &y_parameter_offset,
        &z_parameter_offset,
        const_cast<int**>(&selected),
        const_cast<int**>(&offsets),
        const_cast<int**>(&slot_a),
        const_cast<int**>(&slot_b),
        const_cast<int**>(&edges),
        &phase_count};
    SAD_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(kernel),
        dim3(grid_size),
        dim3(kForwardBlockThreads),
        arguments,
        shared_bytes));
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
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s QUBITS baseline|integrated ITERATIONS\n",
                     argv[0]);
        return 2;
    }
    const int qubits = std::stoi(argv[1]);
    const std::string mode = argv[2];
    const int iterations = std::stoi(argv[3]);
    if (qubits < 4 || (qubits & 1) ||
        (mode != "baseline" && mode != "integrated") || iterations <= 0) {
        return 2;
    }

    std::vector<int> even_selected;
    std::vector<int> even_counts;
    std::vector<int> odd_selected;
    std::vector<int> odd_counts;
    build_bond_phase_maps(qubits,
                          kForwardTileBits,
                          0,
                          &even_selected,
                          &even_counts);
    build_bond_phase_maps(qubits,
                          kForwardTileBits,
                          1,
                          &odd_selected,
                          &odd_counts);
    const auto integrated = build_integrated_maps(qubits);
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
    RotationCoefficients<double>* coefficients = nullptr;
    SAD_CUDA_CHECK(cudaMalloc(&state, state_bytes));
    SAD_CUDA_CHECK(cudaMalloc(&coefficients,
                              3 * qubits * sizeof(*coefficients)));
    SAD_CUDA_CHECK(cudaMemset(state, 0, state_bytes));
    const Complex<double> one = make_complex<double>(1.0, 0.0);
    SAD_CUDA_CHECK(cudaMemcpy(state,
                              &one,
                              sizeof(one),
                              cudaMemcpyHostToDevice));
    std::vector<RotationCoefficients<double>> host_coefficients(
        3 * qubits, {0.123, 0.99240677});
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
    auto launch = [&](Complex<double>* target) {
        if (mode == "baseline") {
            launch_baseline(target);
        } else {
            launch_integrated_schedule(target);
        }
    };

    double max_error = std::numeric_limits<double>::quiet_NaN();
    if (mode == "integrated" && qubits <= 20) {
        Complex<double>* reference = nullptr;
        SAD_CUDA_CHECK(cudaMalloc(&reference, state_bytes));
        SAD_CUDA_CHECK(cudaMemset(reference, 0, state_bytes));
        SAD_CUDA_CHECK(cudaMemcpy(reference,
                                  &one,
                                  sizeof(one),
                                  cudaMemcpyHostToDevice));
        launch_baseline(reference);
        launch_integrated_schedule(state);
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
        cudaFree(reference);
        SAD_CUDA_CHECK(cudaMemset(state, 0, state_bytes));
        SAD_CUDA_CHECK(cudaMemcpy(state,
                                  &one,
                                  sizeof(one),
                                  cudaMemcpyHostToDevice));
    }

    for (int warmup = 0; warmup < 3; ++warmup) launch(state);
    SAD_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    SAD_CUDA_CHECK(cudaEventCreate(&start));
    SAD_CUDA_CHECK(cudaEventCreate(&stop));
    SAD_CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) launch(state);
    SAD_CUDA_CHECK(cudaEventRecord(stop));
    SAD_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    SAD_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::printf("%s,%d,%d,%zu,%zu,%.9f,%.17g\n",
                mode.c_str(),
                qubits,
                kForwardTileBits,
                even_counts.size() + odd_counts.size(),
                integrated.offsets.size() - 1,
                elapsed_ms / iterations,
                max_error);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(state);
    cudaFree(coefficients);
    return 0;
}
