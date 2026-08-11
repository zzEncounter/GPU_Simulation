#include "kernels/xxz.cuh"
#include "runtime/lookups.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace sad;

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "usage: %s QUBITS forward|backward PARITY ITERATIONS\n",
                     argv[0]);
        return 2;
    }
    const int qubits = std::stoi(argv[1]);
    const std::string direction = argv[2];
    const int parity = std::stoi(argv[3]);
    const int iterations = std::stoi(argv[4]);
    if (qubits < 4 || (qubits & 1) ||
        (direction != "forward" && direction != "backward") ||
        (parity != 0 && parity != 1) ||
        iterations <= 0) {
        return 2;
    }

    const int tile_bits =
        direction == "forward" ? kForwardTileBits : kTileBits;

    std::vector<int> selected;
    std::vector<int> pair_counts;
    build_bond_phase_maps(qubits, tile_bits, parity, &selected, &pair_counts);
    int* device_selected = nullptr;
    int* device_pair_counts = nullptr;
    SAD_CUDA_CHECK(cudaMalloc(&device_selected, selected.size() * sizeof(int)));
    SAD_CUDA_CHECK(
        cudaMalloc(&device_pair_counts, pair_counts.size() * sizeof(int)));
    SAD_CUDA_CHECK(cudaMemcpy(device_selected,
                              selected.data(),
                              selected.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(device_pair_counts,
                              pair_counts.data(),
                              pair_counts.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    const std::uint64_t amplitudes = 1ull << qubits;
    Complex<double>* state = nullptr;
    Complex<double>* lambda = nullptr;
    RotationCoefficients<double>* coefficients = nullptr;
    double* gradients = nullptr;
    SAD_CUDA_CHECK(cudaMalloc(&state, amplitudes * sizeof(*state)));
    SAD_CUDA_CHECK(cudaMalloc(&lambda, amplitudes * sizeof(*lambda)));
    SAD_CUDA_CHECK(cudaMalloc(&coefficients,
                              3 * qubits * sizeof(*coefficients)));
    SAD_CUDA_CHECK(cudaMalloc(&gradients, 3 * qubits * sizeof(*gradients)));
    SAD_CUDA_CHECK(cudaMemset(state, 0, amplitudes * sizeof(*state)));
    SAD_CUDA_CHECK(cudaMemset(lambda, 0, amplitudes * sizeof(*lambda)));
    SAD_CUDA_CHECK(cudaMemset(gradients, 0, 3 * qubits * sizeof(*gradients)));
    std::vector<RotationCoefficients<double>> host_coefficients(
        3 * qubits, {0.123, 0.99240677});
    SAD_CUDA_CHECK(cudaMemcpy(coefficients,
                              host_coefficients.data(),
                              host_coefficients.size() * sizeof(*coefficients),
                              cudaMemcpyHostToDevice));

    cudaDeviceProp properties{};
    SAD_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    cudaFuncAttributes attributes{};
    int active_blocks = 0;
    std::size_t dynamic_shared_bytes = 0;
    if (direction == "forward") {
        dynamic_shared_bytes =
            kForwardTileAmplitudes * sizeof(Complex<double>);
        const auto kernel = xxz_matching_forward_kernel<double>;
        SAD_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(dynamic_shared_bytes)));
        SAD_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
        SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            kernel,
            kForwardBlockThreads,
            dynamic_shared_bytes));
    } else {
        const auto kernel = xxz_matching_backward_kernel<double>;
        SAD_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
        SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, kernel, kBlockThreads, 0));
    }

    auto launch = [&]() {
        if (direction == "forward") {
            launch_xxz_matching_forward(
                state,
                coefficients,
                qubits,
                0,
                qubits,
                2 * qubits,
                device_selected,
                device_pair_counts,
                static_cast<int>(pair_counts.size()),
                properties.multiProcessorCount);
        } else {
            launch_xxz_matching_backward(
                state,
                lambda,
                coefficients,
                gradients,
                qubits,
                0,
                qubits,
                2 * qubits,
                device_selected,
                device_pair_counts,
                static_cast<int>(pair_counts.size()),
                properties.multiProcessorCount);
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

    std::printf("%s,%d,%d,%d,%d,%d,%zu,%zu,%zu,%d,%d,%.9f\n",
                direction.c_str(),
                qubits,
                parity,
                direction == "forward" ? kForwardBlockThreads : kBlockThreads,
                direction == "forward" ? kForwardRegisterAmplitudes
                                           : kRegisterAmplitudes,
                tile_bits,
                pair_counts.size(),
                attributes.sharedSizeBytes,
                dynamic_shared_bytes,
                attributes.numRegs,
                active_blocks,
                elapsed_ms / iterations);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(state);
    cudaFree(lambda);
    cudaFree(coefficients);
    cudaFree(gradients);
    cudaFree(device_selected);
    cudaFree(device_pair_counts);
    return 0;
}
