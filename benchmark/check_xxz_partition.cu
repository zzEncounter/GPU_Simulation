#include "kernels/xxz.cuh"
#include "runtime/lookups.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace sad;

namespace {

struct DevicePlan {
    int* selected = nullptr;
    int* pair_counts = nullptr;
    int phases = 0;
};

double difference(Complex<double> lhs, Complex<double> rhs) {
    return std::max(std::abs(lhs.real - rhs.real),
                    std::abs(lhs.imag - rhs.imag));
}

void build_partition(int qubits,
                     int parity,
                     const std::vector<int>& counts,
                     std::vector<int>* selected) {
    const int tile_bits = kTileBits;
    selected->assign(counts.size() * tile_bits, -1);
    int bond = 0;
    for (std::size_t phase = 0; phase < counts.size(); ++phase) {
        int filled = 0;
        for (int offset = 0; offset < counts[phase]; ++offset, ++bond) {
            const int left = parity + 2 * bond;
            (*selected)[phase * tile_bits + filled++] = left;
            (*selected)[phase * tile_bits + filled++] = (left + 1) % qubits;
        }
        for (int qubit = 0; filled < tile_bits && qubit < qubits; ++qubit) {
            bool used = false;
            for (int slot = 0; slot < filled; ++slot) {
                used |= (*selected)[phase * tile_bits + slot] == qubit;
            }
            if (!used) (*selected)[phase * tile_bits + filled++] = qubit;
        }
    }
}

DevicePlan copy_plan(const std::vector<int>& selected,
                     const std::vector<int>& counts) {
    DevicePlan plan;
    plan.phases = static_cast<int>(counts.size());
    SAD_CUDA_CHECK(cudaMalloc(&plan.selected,
                              selected.size() * sizeof(*plan.selected)));
    SAD_CUDA_CHECK(cudaMalloc(&plan.pair_counts,
                              counts.size() * sizeof(*plan.pair_counts)));
    SAD_CUDA_CHECK(cudaMemcpy(plan.selected,
                              selected.data(),
                              selected.size() * sizeof(*plan.selected),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(plan.pair_counts,
                              counts.data(),
                              counts.size() * sizeof(*plan.pair_counts),
                              cudaMemcpyHostToDevice));
    return plan;
}

void free_plan(DevicePlan plan) {
    cudaFree(plan.selected);
    cudaFree(plan.pair_counts);
}

bool check_parity(int parity,
                  double* maximum_state_error,
                  double* maximum_gradient_error) {
    constexpr int qubits = 12;
    constexpr std::uint64_t state_size = 1ull << qubits;
    static_assert(kForwardTileBits == 8 && kTileBits == 8,
                  "compile this check with t64r2");

    std::vector<int> canonical_selected;
    std::vector<int> canonical_counts;
    build_bond_phase_maps(qubits,
                          kTileBits,
                          parity,
                          &canonical_selected,
                          &canonical_counts);
    const std::vector<int> nonuniform_counts{3, 3};
    std::vector<int> nonuniform_selected;
    build_partition(qubits,
                    parity,
                    nonuniform_counts,
                    &nonuniform_selected);
    const DevicePlan canonical =
        copy_plan(canonical_selected, canonical_counts);
    const DevicePlan nonuniform =
        copy_plan(nonuniform_selected, nonuniform_counts);

    std::vector<Complex<double>> phi_input(state_size);
    std::vector<Complex<double>> lambda_input(state_size);
    for (std::uint64_t index = 0; index < state_size; ++index) {
        phi_input[index] = {
            std::sin(0.013 * static_cast<double>(index + 1)),
            std::cos(0.017 * static_cast<double>(index + 3)),
        };
        lambda_input[index] = {
            std::cos(0.019 * static_cast<double>(index + 5)),
            std::sin(0.023 * static_cast<double>(index + 7)),
        };
    }
    std::vector<RotationCoefficients<double>> host_coefficients(3 * qubits);
    for (int parameter = 0; parameter < 3 * qubits; ++parameter) {
        const double half_angle = 0.007 * static_cast<double>(parameter + 1);
        host_coefficients[parameter] = {
            std::sin(half_angle), std::cos(half_angle)};
    }

    Complex<double>* canonical_phi = nullptr;
    Complex<double>* nonuniform_phi = nullptr;
    Complex<double>* canonical_lambda = nullptr;
    Complex<double>* nonuniform_lambda = nullptr;
    RotationCoefficients<double>* coefficients = nullptr;
    double* canonical_gradients = nullptr;
    double* nonuniform_gradients = nullptr;
    SAD_CUDA_CHECK(cudaMalloc(&canonical_phi,
                              state_size * sizeof(*canonical_phi)));
    SAD_CUDA_CHECK(cudaMalloc(&nonuniform_phi,
                              state_size * sizeof(*nonuniform_phi)));
    SAD_CUDA_CHECK(cudaMalloc(&canonical_lambda,
                              state_size * sizeof(*canonical_lambda)));
    SAD_CUDA_CHECK(cudaMalloc(&nonuniform_lambda,
                              state_size * sizeof(*nonuniform_lambda)));
    SAD_CUDA_CHECK(cudaMalloc(&coefficients,
                              3 * qubits * sizeof(*coefficients)));
    SAD_CUDA_CHECK(cudaMalloc(&canonical_gradients,
                              3 * qubits * sizeof(*canonical_gradients)));
    SAD_CUDA_CHECK(cudaMalloc(&nonuniform_gradients,
                              3 * qubits * sizeof(*nonuniform_gradients)));
    SAD_CUDA_CHECK(cudaMemcpy(coefficients,
                              host_coefficients.data(),
                              3 * qubits * sizeof(*coefficients),
                              cudaMemcpyHostToDevice));
    cudaDeviceProp properties{};
    SAD_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    auto copy_state = [&](Complex<double>* destination,
                          const std::vector<Complex<double>>& source) {
        SAD_CUDA_CHECK(cudaMemcpy(destination,
                                  source.data(),
                                  state_size * sizeof(*destination),
                                  cudaMemcpyHostToDevice));
    };
    auto launch_forward = [&](Complex<double>* state, DevicePlan plan) {
        launch_xxz_matching_forward(state,
                                    coefficients,
                                    qubits,
                                    0,
                                    qubits,
                                    2 * qubits,
                                    plan.selected,
                                    plan.pair_counts,
                                    plan.phases,
                                    properties.multiProcessorCount);
    };
    copy_state(canonical_phi, phi_input);
    copy_state(nonuniform_phi, phi_input);
    launch_forward(canonical_phi, canonical);
    launch_forward(nonuniform_phi, nonuniform);
    SAD_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Complex<double>> canonical_state(state_size);
    std::vector<Complex<double>> nonuniform_state(state_size);
    SAD_CUDA_CHECK(cudaMemcpy(canonical_state.data(),
                              canonical_phi,
                              state_size * sizeof(*canonical_phi),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(nonuniform_state.data(),
                              nonuniform_phi,
                              state_size * sizeof(*nonuniform_phi),
                              cudaMemcpyDeviceToHost));
    for (std::uint64_t index = 0; index < state_size; ++index) {
        *maximum_state_error = std::max(
            *maximum_state_error,
            difference(canonical_state[index], nonuniform_state[index]));
    }

    copy_state(canonical_phi, phi_input);
    copy_state(nonuniform_phi, phi_input);
    copy_state(canonical_lambda, lambda_input);
    copy_state(nonuniform_lambda, lambda_input);
    SAD_CUDA_CHECK(cudaMemset(canonical_gradients,
                              0,
                              3 * qubits * sizeof(*canonical_gradients)));
    SAD_CUDA_CHECK(cudaMemset(nonuniform_gradients,
                              0,
                              3 * qubits * sizeof(*nonuniform_gradients)));
    auto launch_backward = [&](Complex<double>* phi,
                               Complex<double>* lambda,
                               double* gradients,
                               DevicePlan plan) {
        launch_xxz_matching_backward(phi,
                                     lambda,
                                     coefficients,
                                     gradients,
                                     qubits,
                                     0,
                                     qubits,
                                     2 * qubits,
                                     plan.selected,
                                     plan.pair_counts,
                                     plan.phases,
                                     properties.multiProcessorCount);
    };
    launch_backward(canonical_phi,
                    canonical_lambda,
                    canonical_gradients,
                    canonical);
    launch_backward(nonuniform_phi,
                    nonuniform_lambda,
                    nonuniform_gradients,
                    nonuniform);
    SAD_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Complex<double>> canonical_backward_phi(state_size);
    std::vector<Complex<double>> nonuniform_backward_phi(state_size);
    std::vector<Complex<double>> canonical_backward_lambda(state_size);
    std::vector<Complex<double>> nonuniform_backward_lambda(state_size);
    std::vector<double> canonical_gradient(3 * qubits);
    std::vector<double> nonuniform_gradient(3 * qubits);
    SAD_CUDA_CHECK(cudaMemcpy(canonical_backward_phi.data(),
                              canonical_phi,
                              state_size * sizeof(*canonical_phi),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(nonuniform_backward_phi.data(),
                              nonuniform_phi,
                              state_size * sizeof(*nonuniform_phi),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(canonical_backward_lambda.data(),
                              canonical_lambda,
                              state_size * sizeof(*canonical_lambda),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(nonuniform_backward_lambda.data(),
                              nonuniform_lambda,
                              state_size * sizeof(*nonuniform_lambda),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(canonical_gradient.data(),
                              canonical_gradients,
                              3 * qubits * sizeof(*canonical_gradients),
                              cudaMemcpyDeviceToHost));
    SAD_CUDA_CHECK(cudaMemcpy(nonuniform_gradient.data(),
                              nonuniform_gradients,
                              3 * qubits * sizeof(*nonuniform_gradients),
                              cudaMemcpyDeviceToHost));
    for (std::uint64_t index = 0; index < state_size; ++index) {
        *maximum_state_error = std::max(
            {*maximum_state_error,
             difference(canonical_backward_phi[index],
                        nonuniform_backward_phi[index]),
             difference(canonical_backward_lambda[index],
                        nonuniform_backward_lambda[index])});
    }
    for (int parameter = 0; parameter < 3 * qubits; ++parameter) {
        *maximum_gradient_error = std::max(
            *maximum_gradient_error,
            std::abs(canonical_gradient[parameter] -
                     nonuniform_gradient[parameter]));
    }

    cudaFree(canonical_phi);
    cudaFree(nonuniform_phi);
    cudaFree(canonical_lambda);
    cudaFree(nonuniform_lambda);
    cudaFree(coefficients);
    cudaFree(canonical_gradients);
    cudaFree(nonuniform_gradients);
    free_plan(canonical);
    free_plan(nonuniform);
    return *maximum_state_error <= 1e-12 &&
           *maximum_gradient_error <= 1e-8;
}

}  // namespace

int main() {
    double maximum_state_error = 0;
    double maximum_gradient_error = 0;
    const bool even_ok = check_parity(
        0, &maximum_state_error, &maximum_gradient_error);
    const bool odd_ok = check_parity(
        1, &maximum_state_error, &maximum_gradient_error);
    std::printf("XXZ partition correctness: state=%.3e gradient=%.3e\n",
                maximum_state_error,
                maximum_gradient_error);
    return even_ok && odd_ok ? 0 : 1;
}
