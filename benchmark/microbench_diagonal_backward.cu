#include "kernels/diagonal.cuh"
#include "runtime/lookups.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace sad;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s QUBITS rz|combined|qaoa ITERATIONS\n",
                     argv[0]);
        return 2;
    }
    const int qubits = std::stoi(argv[1]);
    const std::string strategy = argv[2];
    const int iterations = std::stoi(argv[3]);
    if (qubits < 4 || (qubits & 1) || qubits > kMaxQubits ||
        (strategy != "rz" && strategy != "combined" &&
         strategy != "qaoa") ||
        iterations <= 0) {
        return 2;
    }

    const uint64_t state_size = 1ull << qubits;
    std::vector<double> parameters(2 * qubits, 0.123);
    DiagonalLookupData<double> lookup;
    lookup.offsets_by_parameter.assign(2 * qubits, kNoParameterOffset);
    std::size_t rz_offset = 0;
    std::size_t even_offset = 0;
    std::size_t odd_offset = 0;
    if (strategy == "qaoa") {
        if constexpr (kQaoaCompactLookup) {
            append_ring_rzz_compact_lookup_group(
                parameters.data(), 0, qubits, &lookup);
        } else {
            append_shared_diagonal_lookup_group(
                parameters.data(), 0, qubits / 2, &lookup);
        }
    } else {
        append_diagonal_lookup_group(parameters.data(), 0, qubits, &lookup);
        if (strategy == "combined") {
            even_offset = lookup.factors.size();
            append_diagonal_lookup_group(
                parameters.data(), qubits, qubits / 2, &lookup);
            odd_offset = lookup.factors.size();
            append_diagonal_lookup_group(
                parameters.data(), qubits + qubits / 2, qubits / 2, &lookup);
        }
    }

    Complex<double>* phi = nullptr;
    Complex<double>* lambda = nullptr;
    Complex<double>* device_lookup = nullptr;
    double* gradients = nullptr;
    SAD_CUDA_CHECK(cudaMalloc(&phi, state_size * sizeof(*phi)));
    SAD_CUDA_CHECK(cudaMalloc(&lambda, state_size * sizeof(*lambda)));
    SAD_CUDA_CHECK(
        cudaMalloc(&device_lookup, lookup.factors.size() * sizeof(*device_lookup)));
    SAD_CUDA_CHECK(cudaMalloc(&gradients, 2 * qubits * sizeof(*gradients)));
    SAD_CUDA_CHECK(cudaMemset(phi, 0, state_size * sizeof(*phi)));
    SAD_CUDA_CHECK(cudaMemset(lambda, 0, state_size * sizeof(*lambda)));
    SAD_CUDA_CHECK(cudaMemset(gradients, 0, 2 * qubits * sizeof(*gradients)));
    SAD_CUDA_CHECK(cudaMemcpy(device_lookup,
                              lookup.factors.data(),
                              lookup.factors.size() * sizeof(*device_lookup),
                              cudaMemcpyHostToDevice));

    cudaDeviceProp properties{};
    SAD_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    const uint64_t required =
        (state_size + kDiagonalBlockThreads - 1) / kDiagonalBlockThreads;
    const int ordinary_grid = static_cast<int>(std::min<uint64_t>(
        required, static_cast<uint64_t>(properties.multiProcessorCount) * 4));
    auto launch = [&]() {
        if (strategy == "rz") {
            launch_diagonal_backward<double, DiagonalGate::RZ>(
                phi,
                lambda,
                device_lookup + rz_offset,
                gradients,
                state_size,
                qubits,
                0,
                qubits,
                ordinary_grid);
        } else if (strategy == "combined") {
            launch_rz_rzz_backward(phi,
                                   lambda,
                                   device_lookup + rz_offset,
                                   device_lookup + even_offset,
                                   device_lookup + odd_offset,
                                   gradients,
                                   state_size,
                                   qubits,
                                   0,
                                   qubits,
                                   qubits + qubits / 2,
                                   properties.multiProcessorCount);
        } else {
            launch_shared_ring_rzz_backward(phi,
                                            lambda,
                                            device_lookup,
                                            gradients,
                                            state_size,
                                            qubits,
                                            0,
                                            ordinary_grid);
        }
    };

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
    std::printf("%s,%d,%d,%d,%d,%.9f\n",
                strategy.c_str(),
                qubits,
                strategy == "qaoa" ? kSharedDiagonalBlockThreads
                                     : strategy == "combined"
                                           ? kCombinedDiagonalThreads
                                           : kDiagonalBlockThreads,
                static_cast<int>(kLegacyBlockReduction),
                static_cast<int>(kDiagonalWarpAtomic),
                elapsed_ms / iterations);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(phi);
    cudaFree(lambda);
    cudaFree(device_lookup);
    cudaFree(gradients);
    return 0;
}
