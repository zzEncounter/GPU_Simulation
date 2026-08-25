#include "kernels/fused/diagonal.cuh"
#include "runtime/lookups.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace sad;

template <bool Rzz>
__global__ void direct_diagonal_kernel(Complex<double>* state,
                                       const double* parameters,
                                       uint64_t state_size,
                                       int qubits) {
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        double phase = 0;
        for (int gate = 0; gate < qubits; ++gate) {
            int eigenvalue = 0;
            if constexpr (Rzz) {
                const int right = (gate + 1) % qubits;
                eigenvalue = ((index >> gate) & 1ull) ==
                                     ((index >> right) & 1ull)
                                 ? 1
                                 : -1;
            } else {
                eigenvalue = ((index >> gate) & 1ull) ? -1 : 1;
            }
            phase -= 0.5 * parameters[gate] * eigenvalue;
        }
        double sine = 0;
        double cosine = 0;
        sincos(phase, &sine, &cosine);
        state[index] = multiply(state[index], Complex<double>{cosine, sine});
    }
}

template <bool Rzz>
__global__ void lookup_diagonal_kernel(Complex<double>* state,
                                       const Complex<double>* first,
                                       const Complex<double>* second,
                                       uint64_t state_size,
                                       int qubits) {
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        Complex<double> factor;
        if constexpr (Rzz) {
            factor = diagonal_lookup_factor<double, DiagonalGate::RZZ_EVEN>(
                index, first, qubits, qubits / 2);
            factor = multiply(
                factor,
                diagonal_lookup_factor<double, DiagonalGate::RZZ_ODD>(
                    index, second, qubits, qubits / 2));
        } else {
            factor = diagonal_lookup_factor<double, DiagonalGate::RZ>(
                index, first, qubits, qubits);
        }
        state[index] = multiply(state[index], factor);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const int qubits = std::stoi(argv[1]);
    const std::string gate = argv[2];
    const std::string strategy = argv[3];
    const bool rzz = gate == "rzz";
    const uint64_t state_size = 1ull << qubits;
    std::vector<double> parameters(qubits, 0.123);
    DiagonalLookupData<double> lookup;
    lookup.offsets_by_parameter.assign(2 * qubits, kNoParameterOffset);
    const auto build_start = std::chrono::steady_clock::now();
    if (rzz) {
        append_diagonal_lookup_group(
            parameters.data(), 0, qubits / 2, &lookup);
        append_diagonal_lookup_group(parameters.data(),
                                     qubits / 2,
                                     qubits / 2,
                                     &lookup);
    } else {
        append_diagonal_lookup_group(parameters.data(), 0, qubits, &lookup);
    }
    const auto build_stop = std::chrono::steady_clock::now();
    const double build_ms =
        std::chrono::duration<double, std::milli>(build_stop - build_start)
            .count();
    Complex<double>* state = nullptr;
    Complex<double>* device_lookup = nullptr;
    double* device_parameters = nullptr;
    cudaMalloc(&state, state_size * sizeof(*state));
    cudaMalloc(&device_lookup, lookup.factors.size() * sizeof(*device_lookup));
    cudaMalloc(&device_parameters, qubits * sizeof(double));
    cudaMemset(state, 0, state_size * sizeof(*state));
    cudaDeviceSynchronize();
    const auto copy_start = std::chrono::steady_clock::now();
    cudaMemcpy(device_lookup,
               lookup.factors.data(),
               lookup.factors.size() * sizeof(*device_lookup),
               cudaMemcpyHostToDevice);
    const auto copy_stop = std::chrono::steady_clock::now();
    const double copy_ms =
        std::chrono::duration<double, std::milli>(copy_stop - copy_start)
            .count();
    cudaMemcpy(device_parameters,
               parameters.data(),
               qubits * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    const int grid = static_cast<int>(std::min<uint64_t>(
        (state_size + kBlockThreads - 1) / kBlockThreads,
        properties.multiProcessorCount * 4ull));
    const size_t second_offset = lookup.offsets_by_parameter[qubits / 2];
    auto launch = [&]() {
        if (strategy == "direct") {
            if (rzz) {
                direct_diagonal_kernel<true><<<grid, kBlockThreads>>>(
                    state, device_parameters, state_size, qubits);
            } else {
                direct_diagonal_kernel<false><<<grid, kBlockThreads>>>(
                    state, device_parameters, state_size, qubits);
            }
        } else if (rzz) {
            lookup_diagonal_kernel<true><<<grid, kBlockThreads>>>(
                state,
                device_lookup,
                device_lookup + second_offset,
                state_size,
                qubits);
        } else {
            lookup_diagonal_kernel<false><<<grid, kBlockThreads>>>(
                state, device_lookup, nullptr, state_size, qubits);
        }
    };
    for (int warmup = 0; warmup < 5; ++warmup) launch();
    cudaDeviceSynchronize();
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int iterations = 30;
    cudaEventRecord(start);
    for (int iteration = 0; iteration < iterations; ++iteration) launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::printf("%s,%s,%d,%d,%zu,%.9f,%.9f,%.9f\n",
                gate.c_str(),
                strategy.c_str(),
                qubits,
                kDiagonalLookupBits,
                lookup.factors.size() * sizeof(Complex<double>),
                build_ms,
                copy_ms,
                elapsed_ms / iterations);
    return 0;
}
