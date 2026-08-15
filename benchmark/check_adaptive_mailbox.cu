#include "kernels/diagonal.cuh"
#include "kernels/rotation.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace sad;

namespace {

double difference(Complex<double> lhs, Complex<double> rhs) {
    return std::max(std::abs(lhs.real - rhs.real),
                    std::abs(lhs.imag - rhs.imag));
}

template <NonDiagonalGate Gate>
bool check_gate(double* maximum_state_error,
                double* maximum_gradient_error) {
    constexpr int qubits = 12;
    constexpr uint64_t state_size = 1ull << qubits;
    static_assert(kForwardTileBits == 10 && kTileBits == 10,
                  "compile this check with t64r4");

    std::vector<int> selected(2 * kTileBits);
    std::vector<int> masks{(1 << kTileBits) - 1, 3};
    for (int slot = 0; slot < kTileBits; ++slot) selected[slot] = slot;
    selected[kTileBits] = 10;
    selected[kTileBits + 1] = 11;
    for (int slot = 2; slot < kTileBits; ++slot) {
        selected[kTileBits + slot] = slot - 2;
    }
    const uint64_t warp_phase_mask = 1;

    std::vector<Complex<double>> phi_input(state_size);
    std::vector<Complex<double>> lambda_input(state_size);
    for (uint64_t index = 0; index < state_size; ++index) {
        phi_input[index] = {
            std::sin(0.013 * static_cast<double>(index + 1)),
            std::cos(0.017 * static_cast<double>(index + 3)),
        };
        lambda_input[index] = {
            std::cos(0.019 * static_cast<double>(index + 5)),
            std::sin(0.023 * static_cast<double>(index + 7)),
        };
    }
    std::vector<RotationCoefficients<double>> host_coefficients(qubits);
    for (int qubit = 0; qubit < qubits; ++qubit) {
        const double half_angle = 0.031 * static_cast<double>(qubit + 1);
        host_coefficients[qubit] = {
            std::sin(half_angle), std::cos(half_angle)};
    }

    Complex<double>* full_phi = nullptr;
    Complex<double>* adaptive_phi = nullptr;
    Complex<double>* full_lambda = nullptr;
    Complex<double>* adaptive_lambda = nullptr;
    RotationCoefficients<double>* coefficients = nullptr;
    double* full_gradients = nullptr;
    double* adaptive_gradients = nullptr;
    int* device_selected = nullptr;
    int* device_masks = nullptr;
    SAD_CUDA_CHECK(cudaMalloc(&full_phi, state_size * sizeof(*full_phi)));
    SAD_CUDA_CHECK(cudaMalloc(
        &adaptive_phi, state_size * sizeof(*adaptive_phi)));
    SAD_CUDA_CHECK(cudaMalloc(&full_lambda,
                              state_size * sizeof(*full_lambda)));
    SAD_CUDA_CHECK(cudaMalloc(&adaptive_lambda,
                              state_size * sizeof(*adaptive_lambda)));
    SAD_CUDA_CHECK(cudaMalloc(
        &coefficients, qubits * sizeof(*coefficients)));
    SAD_CUDA_CHECK(cudaMalloc(
        &full_gradients, qubits * sizeof(*full_gradients)));
    SAD_CUDA_CHECK(cudaMalloc(
        &adaptive_gradients, qubits * sizeof(*adaptive_gradients)));
    SAD_CUDA_CHECK(cudaMalloc(
        &device_selected, selected.size() * sizeof(*device_selected)));
    SAD_CUDA_CHECK(cudaMalloc(
        &device_masks, masks.size() * sizeof(*device_masks)));
    SAD_CUDA_CHECK(cudaMemcpy(coefficients,
                              host_coefficients.data(),
                              qubits * sizeof(*coefficients),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(device_selected,
                              selected.data(),
                              selected.size() * sizeof(*device_selected),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(device_masks,
                              masks.data(),
                              masks.size() * sizeof(*device_masks),
                              cudaMemcpyHostToDevice));
    cudaDeviceProp properties{};
    SAD_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    auto copy_input = [&](Complex<double>* destination,
                          const std::vector<Complex<double>>& source) {
        SAD_CUDA_CHECK(cudaMemcpy(destination,
                                  source.data(),
                                  state_size * sizeof(*destination),
                                  cudaMemcpyHostToDevice));
    };
    copy_input(full_phi, phi_input);
    copy_input(adaptive_phi, phi_input);
    launch_non_diagonal_forward<double, Gate>(
        full_phi, coefficients, qubits, 0, device_selected, device_masks, 2,
        properties.multiProcessorCount, false, ~0ull);
    launch_non_diagonal_forward<double, Gate>(
        adaptive_phi, coefficients, qubits, 0, device_selected, device_masks,
        2, properties.multiProcessorCount, false, warp_phase_mask);
    SAD_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<Complex<double>> full_state(state_size);
    std::vector<Complex<double>> adaptive_state(state_size);
    SAD_CUDA_CHECK(cudaMemcpy(full_state.data(),
                              full_phi,
                              state_size * sizeof(*full_phi),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(adaptive_state.data(),
                              adaptive_phi,
                              state_size * sizeof(*adaptive_phi),
                              cudaMemcpyDeviceToHost));
    for (uint64_t index = 0; index < state_size; ++index) {
        *maximum_state_error = std::max(
            *maximum_state_error,
            difference(full_state[index], adaptive_state[index]));
    }

    copy_input(full_phi, phi_input);
    copy_input(adaptive_phi, phi_input);
    copy_input(full_lambda, lambda_input);
    copy_input(adaptive_lambda, lambda_input);
    SAD_CUDA_CHECK(cudaMemset(full_gradients, 0,
                              qubits * sizeof(*full_gradients)));
    SAD_CUDA_CHECK(cudaMemset(adaptive_gradients, 0,
                              qubits * sizeof(*adaptive_gradients)));
    launch_non_diagonal_backward<double, Gate>(
        full_phi, full_lambda, coefficients, full_gradients, qubits, 0,
        device_selected, device_masks, 2, properties.multiProcessorCount,
        false, ~0ull);
    launch_non_diagonal_backward<double, Gate>(
        adaptive_phi, adaptive_lambda, coefficients, adaptive_gradients,
        qubits, 0, device_selected, device_masks, 2,
        properties.multiProcessorCount, false, warp_phase_mask);
    SAD_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<Complex<double>> full_backward_phi(state_size);
    std::vector<Complex<double>> adaptive_backward_phi(state_size);
    std::vector<Complex<double>> full_backward_lambda(state_size);
    std::vector<Complex<double>> adaptive_backward_lambda(state_size);
    std::vector<double> full_gradient(qubits);
    std::vector<double> adaptive_gradient(qubits);
    SAD_CUDA_CHECK(cudaMemcpy(full_backward_phi.data(), full_phi,
                              state_size * sizeof(*full_phi),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(adaptive_backward_phi.data(), adaptive_phi,
                              state_size * sizeof(*adaptive_phi),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(full_backward_lambda.data(), full_lambda,
                              state_size * sizeof(*full_lambda),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(adaptive_backward_lambda.data(), adaptive_lambda,
                              state_size * sizeof(*adaptive_lambda),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(full_gradient.data(), full_gradients,
                              qubits * sizeof(*full_gradients),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(adaptive_gradient.data(), adaptive_gradients,
                              qubits * sizeof(*adaptive_gradients),
                              cudaMemcpyDeviceToHost));
    for (uint64_t index = 0; index < state_size; ++index) {
        *maximum_state_error = std::max(
            {*maximum_state_error,
             difference(full_backward_phi[index],
                        adaptive_backward_phi[index]),
             difference(full_backward_lambda[index],
                        adaptive_backward_lambda[index])});
    }
    for (int qubit = 0; qubit < qubits; ++qubit) {
        *maximum_gradient_error = std::max(
            *maximum_gradient_error,
            std::abs(full_gradient[qubit] - adaptive_gradient[qubit]));
    }

    cudaFree(full_phi);
    cudaFree(adaptive_phi);
    cudaFree(full_lambda);
    cudaFree(adaptive_lambda);
    cudaFree(coefficients);
    cudaFree(full_gradients);
    cudaFree(adaptive_gradients);
    cudaFree(device_selected);
    cudaFree(device_masks);
    return *maximum_state_error <= 1e-12 &&
           *maximum_gradient_error <= 1e-9;
}

}  // namespace

int main() {
    double maximum_state_error = 0;
    double maximum_gradient_error = 0;
    const bool rx_ok = check_gate<NonDiagonalGate::RX>(
        &maximum_state_error, &maximum_gradient_error);
    const bool ry_ok = check_gate<NonDiagonalGate::RY>(
        &maximum_state_error, &maximum_gradient_error);
    std::printf("adaptive mailbox correctness: state=%.3e gradient=%.3e\n",
                maximum_state_error, maximum_gradient_error);
    return rx_ok && ry_ok ? 0 : 1;
}
