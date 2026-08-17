#include "kernels/ring_cnot.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace sad;

template <typename T>
__global__ void ring_cnot_scatter_kernel(const Complex<T>* input,
                                         Complex<T>* output,
                                         const Complex<T>* lambda_input,
                                         Complex<T>* lambda_output,
                                         std::uint64_t state_size,
                                         int qubits,
                                         bool adjoint) {
    for (std::uint64_t input_index =
             blockIdx.x * blockDim.x + threadIdx.x;
         input_index < state_size;
         input_index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const std::uint64_t output_index =
            adjoint ? apply_ring_cnot_inverse_to_basis(input_index, qubits)
                    : apply_ring_cnot_forward_to_basis(input_index, qubits);
        output[output_index] = input[input_index];
        if (lambda_input != nullptr) {
            lambda_output[output_index] = lambda_input[input_index];
        }
    }
}

template <typename T>
__global__ void state_copy_kernel(const Complex<T>* input,
                                  Complex<T>* output,
                                  std::uint64_t state_size) {
    for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        output[index] = input[index];
    }
}

__device__ __forceinline__ std::uint64_t insert_zero_bit(
    std::uint64_t value, int bit) {
    const std::uint64_t lower_mask = (1ull << bit) - 1;
    return (value & lower_mask) | ((value & ~lower_mask) << 1);
}

template <typename T>
__global__ void cnot_in_place_kernel(Complex<T>* state,
                                     std::uint64_t pair_count,
                                     int control,
                                     int target) {
    const int low = min(control, target);
    const int high = max(control, target);
    for (std::uint64_t pair = blockIdx.x * blockDim.x + threadIdx.x;
         pair < pair_count;
         pair += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        std::uint64_t zero_target = insert_zero_bit(pair, low);
        zero_target = insert_zero_bit(zero_target, high);
        zero_target |= 1ull << control;
        const std::uint64_t one_target = zero_target | (1ull << target);
        const Complex<T> zero = state[zero_target];
        const Complex<T> one = state[one_target];
        state[zero_target] = one;
        state[one_target] = zero;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s QUBITS copy|gather|scatter|gather-adjoint|"
                     "scatter-adjoint|dual|dual-scatter|dual-adjoint|"
                     "dual-scatter-adjoint|inplace "
                     "ITERATIONS\n",
                     argv[0]);
        return 2;
    }
    const int qubits = std::stoi(argv[1]);
    const std::string mode = argv[2];
    const int iterations = std::stoi(argv[3]);
    if (qubits < 2 || qubits >= 31 || iterations <= 0 ||
        (mode != "copy" && mode != "gather" && mode != "scatter" &&
         mode != "gather-adjoint" && mode != "scatter-adjoint" &&
         mode != "dual" && mode != "dual-scatter" &&
         mode != "dual-adjoint" && mode != "dual-scatter-adjoint" &&
         mode != "inplace")) {
        return 2;
    }

    cudaDeviceProp properties{};
    SAD_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    const std::uint64_t amplitudes = 1ull << qubits;
    const std::size_t state_bytes = amplitudes * sizeof(Complex<double>);
    const int grid = static_cast<int>(std::min<std::uint64_t>(
        (amplitudes + kOrdinaryBlockThreads - 1) / kOrdinaryBlockThreads,
        properties.multiProcessorCount * 4ull));
    const int pair_grid = static_cast<int>(std::min<std::uint64_t>(
        ((amplitudes >> 2) + kOrdinaryBlockThreads - 1) /
            kOrdinaryBlockThreads,
        properties.multiProcessorCount * 4ull));

    Complex<double>* first = nullptr;
    Complex<double>* second = nullptr;
    Complex<double>* third = nullptr;
    Complex<double>* fourth = nullptr;
    SAD_CUDA_CHECK(cudaMalloc(&first, state_bytes));
    SAD_CUDA_CHECK(cudaMalloc(&second, state_bytes));
    const bool dual = mode.rfind("dual", 0) == 0;
    const bool adjoint = mode.find("adjoint") != std::string::npos;
    if (dual) {
        SAD_CUDA_CHECK(cudaMalloc(&third, state_bytes));
        SAD_CUDA_CHECK(cudaMalloc(&fourth, state_bytes));
    }
    SAD_CUDA_CHECK(cudaMemset(first, 0, state_bytes));
    SAD_CUDA_CHECK(cudaMemset(second, 0, state_bytes));
    if (third != nullptr) {
        SAD_CUDA_CHECK(cudaMemset(third, 0, state_bytes));
        SAD_CUDA_CHECK(cudaMemset(fourth, 0, state_bytes));
    }
    const Complex<double> marker = make_complex<double>(0.25, -0.75);
    const std::uint64_t marker_index = std::min<std::uint64_t>(37, amplitudes - 1);
    SAD_CUDA_CHECK(cudaMemcpy(first + marker_index,
                              &marker,
                              sizeof(marker),
                              cudaMemcpyHostToDevice));
    if (third != nullptr) {
        SAD_CUDA_CHECK(cudaMemcpy(third + marker_index,
                                  &marker,
                                  sizeof(marker),
                                  cudaMemcpyHostToDevice));
    }

    StatePair<double> phi{first, second};
    StatePair<double> lambda{third, fourth};
    auto launch = [&]() {
        if (mode == "gather" || mode == "gather-adjoint") {
            launch_cnot(&phi,
                        static_cast<StatePair<double>*>(nullptr),
                        amplitudes,
                        qubits,
                        adjoint,
                        grid);
        } else if (mode == "dual" || mode == "dual-adjoint") {
            launch_cnot(&phi,
                        &lambda,
                        amplitudes,
                        qubits,
                        adjoint,
                        grid);
        } else if (mode == "dual-scatter" ||
                   mode == "dual-scatter-adjoint") {
            ring_cnot_scatter_kernel<<<grid, kOrdinaryBlockThreads>>>(
                phi.current,
                phi.scratch,
                lambda.current,
                lambda.scratch,
                amplitudes,
                qubits,
                adjoint);
            SAD_CUDA_CHECK(cudaGetLastError());
            phi.swap();
            lambda.swap();
        } else if (mode == "scatter" || mode == "scatter-adjoint") {
            ring_cnot_scatter_kernel<<<grid, kOrdinaryBlockThreads>>>(
                phi.current,
                phi.scratch,
                static_cast<const Complex<double>*>(nullptr),
                static_cast<Complex<double>*>(nullptr),
                amplitudes,
                qubits,
                adjoint);
            SAD_CUDA_CHECK(cudaGetLastError());
            phi.swap();
        } else if (mode == "copy") {
            state_copy_kernel<<<grid, kOrdinaryBlockThreads>>>(
                phi.current, phi.scratch, amplitudes);
            SAD_CUDA_CHECK(cudaGetLastError());
            phi.swap();
        } else {
            for (int control = 0; control < qubits; ++control) {
                cnot_in_place_kernel<<<pair_grid, kOrdinaryBlockThreads>>>(
                    phi.current,
                    amplitudes >> 2,
                    control,
                    (control + 1) % qubits);
            }
            SAD_CUDA_CHECK(cudaGetLastError());
        }
    };

    double verification_max_error =
        std::numeric_limits<double>::quiet_NaN();
    if ((mode == "scatter" || mode == "scatter-adjoint" ||
         mode == "inplace") &&
        qubits <= 20) {
        Complex<double>* reference = nullptr;
        Complex<double>* reference_scratch = nullptr;
        SAD_CUDA_CHECK(cudaMalloc(&reference, state_bytes));
        SAD_CUDA_CHECK(cudaMalloc(&reference_scratch, state_bytes));
        SAD_CUDA_CHECK(cudaMemset(reference, 0, state_bytes));
        SAD_CUDA_CHECK(cudaMemcpy(reference + marker_index,
                                  &marker,
                                  sizeof(marker),
                                  cudaMemcpyHostToDevice));
        StatePair<double> reference_pair{reference, reference_scratch};
        launch_cnot(&reference_pair,
                    static_cast<StatePair<double>*>(nullptr),
                    amplitudes,
                    qubits,
                    adjoint,
                    grid);
        launch();
        SAD_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<Complex<double>> host_reference(amplitudes);
        std::vector<Complex<double>> host_candidate(amplitudes);
        SAD_CUDA_CHECK(cudaMemcpy(host_reference.data(),
                                  reference_pair.current,
                                  state_bytes,
                                  cudaMemcpyDeviceToHost));
        SAD_CUDA_CHECK(cudaMemcpy(host_candidate.data(),
                                  phi.current,
                                  state_bytes,
                                  cudaMemcpyDeviceToHost));
        verification_max_error = 0.0;
        for (std::uint64_t index = 0; index < amplitudes; ++index) {
            verification_max_error = std::max(
                verification_max_error,
                std::hypot(host_reference[index].real -
                               host_candidate[index].real,
                           host_reference[index].imag -
                               host_candidate[index].imag));
        }
        cudaFree(reference);
        cudaFree(reference_scratch);
        SAD_CUDA_CHECK(cudaMemset(first, 0, state_bytes));
        SAD_CUDA_CHECK(cudaMemset(second, 0, state_bytes));
        SAD_CUDA_CHECK(cudaMemcpy(first + marker_index,
                                  &marker,
                                  sizeof(marker),
                                  cudaMemcpyHostToDevice));
        phi = {first, second};
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
    const double average_ms = elapsed_ms / iterations;
    const double logical_bytes =
        2.0 * state_bytes *
        (mode == "inplace"
             ? qubits
             : (dual ? 2 : 1));
    const double effective_gib_per_second =
        logical_bytes / (average_ms * 1.0e-3) / static_cast<double>(1ull << 30);
    std::printf("%s,%d,%zu,%.9f,%.6f,%.17g\n",
                mode.c_str(),
                qubits,
                state_bytes,
                average_ms,
                effective_gib_per_second,
                verification_max_error);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(first);
    cudaFree(second);
    cudaFree(third);
    cudaFree(fourth);
    return 0;
}
