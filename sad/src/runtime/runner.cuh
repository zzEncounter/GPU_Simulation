#pragma once

#include "sad_api.h"

#include "../kernels/hamiltonian.cuh"
#include "circuit_execution.cuh"
#include "workspace.cuh"

#include <cstddef>

namespace sad {

template <typename T>
void run_step(PreparedWorkspace<T>* workspace,
              HostResults<T>* host_results,
              double* forward_time,
              double* hamiltonian_time,
              double* backward_time) {
    const auto& config = workspace->config;
    StatePair<T> phi{workspace->phi_a.get(), workspace->phi_b.get()};
    StatePair<T> lambda{workspace->lambda_a.get(), workspace->lambda_b.get()};

    const double measured_forward = host_results->timer.measure([&]() {
        run_forward(config.circuit,
                    config.qubits,
                    config.layers,
                    workspace->state_size,
                    workspace->rotation_coefficients.get(),
                    workspace->initial_state_lookup.get(),
                    workspace->diagonal_lookup.get(),
                    workspace->diagonal_lookup_data.offsets_by_parameter.data(),
                    workspace->forward_selected_maps.get(),
                    workspace->forward_target_masks.get(),
                    workspace->forward_phase_count,
                    workspace->forward_xxz_even_selected_maps.get(),
                    workspace->forward_xxz_even_pair_counts.get(),
                    workspace->forward_xxz_even_phase_count,
                    workspace->forward_xxz_odd_selected_maps.get(),
                    workspace->forward_xxz_odd_pair_counts.get(),
                    workspace->forward_xxz_odd_phase_count,
                    workspace->multiprocessors,
                    workspace->ordinary_grid,
                    workspace->execution_mode,
                    &phi);
    });

    const double measured_hamiltonian = host_results->timer.measure([&]() {
        SAD_CUDA_CHECK(cudaMemset(workspace->energy.get(), 0, sizeof(double)));
        if (use_real_amplitude_state(config.circuit,
                                     workspace->execution_mode)) {
            real_hamiltonian_kernel<T>
                <<<workspace->ordinary_grid, kOrdinaryBlockThreads>>>(
                    reinterpret_cast<T*>(phi.current),
                    reinterpret_cast<T*>(lambda.current),
                    workspace->state_size,
                    config.qubits,
                    workspace->energy.get());
        } else if (config.circuit == SAD_CIRCUIT_QAOA ||
                   config.circuit == SAD_CIRCUIT_QAOA_NS) {
            qaoa_cost_hamiltonian_kernel<T>
                <<<workspace->ordinary_grid, kOrdinaryBlockThreads>>>(
                    phi.current,
                    lambda.current,
                    workspace->state_size,
                    config.qubits,
                    workspace->energy.get());
        } else if (config.circuit == SAD_CIRCUIT_XXZ_HVA) {
            xxz_hamiltonian_kernel<T>
                <<<workspace->ordinary_grid, kOrdinaryBlockThreads>>>(
                    phi.current,
                    lambda.current,
                    workspace->state_size,
                    config.qubits,
                    workspace->energy.get());
        } else if (config.circuit == SAD_CIRCUIT_MERA) {
            mera_hamiltonian_kernel<T>
                <<<workspace->ordinary_grid, kOrdinaryBlockThreads>>>(
                    phi.current,
                    lambda.current,
                    workspace->state_size,
                    config.qubits - 1,
                    workspace->energy.get());
        } else if (config.circuit == SAD_CIRCUIT_EQUIVARIANT_QNN) {
            equivariant_qnn_hamiltonian_kernel<T>
                <<<workspace->ordinary_grid, kOrdinaryBlockThreads>>>(
                    phi.current, lambda.current, workspace->state_size,
                    config.qubits, workspace->energy.get());
        } else if (config.circuit == SAD_CIRCUIT_DATA_REUPLOADING) {
            data_reuploading_hamiltonian_kernel<T>
                <<<workspace->ordinary_grid, kOrdinaryBlockThreads>>>(
                    phi.current, lambda.current, workspace->state_size,
                    workspace->energy.get());
        } else {
            hamiltonian_kernel<T>
                <<<workspace->ordinary_grid, kOrdinaryBlockThreads>>>(
                phi.current,
                lambda.current,
                workspace->state_size,
                config.qubits,
                workspace->energy.get());
        }
        SAD_CUDA_CHECK(cudaGetLastError());
        SAD_CUDA_CHECK(cudaMemcpyAsync(host_results->energy.get(),
                                       workspace->energy.get(),
                                       sizeof(double),
                                       cudaMemcpyDeviceToHost));
    });

    const double measured_backward = host_results->timer.measure([&]() {
        SAD_CUDA_CHECK(cudaMemset(workspace->gradients.get(),
                                  0,
                                  config.parameter_count * sizeof(double)));
        run_backward(config.circuit,
                     config.qubits,
                     config.layers,
                     workspace->state_size,
                     workspace->rotation_coefficients.get(),
                     workspace->diagonal_lookup.get(),
                     workspace->diagonal_lookup_data.offsets_by_parameter.data(),
                     workspace->gradients.get(),
                     workspace->backward_selected_maps.get(),
                     workspace->backward_target_masks.get(),
                     workspace->backward_target_phases.get(),
                     workspace->backward_phase_count,
                     workspace->backward_xxz_even_selected_maps.get(),
                     workspace->backward_xxz_even_pair_counts.get(),
                     workspace->backward_xxz_even_phase_count,
                     workspace->backward_xxz_odd_selected_maps.get(),
                     workspace->backward_xxz_odd_pair_counts.get(),
                     workspace->backward_xxz_odd_phase_count,
                     workspace->multiprocessors,
                     workspace->ordinary_grid,
                     workspace->execution_mode,
                     &phi,
                     &lambda);
        SAD_CUDA_CHECK(cudaMemcpyAsync(host_results->gradients.get(),
                                       workspace->gradients.get(),
                                       config.parameter_count * sizeof(double),
                                       cudaMemcpyDeviceToHost));
    });

    if (forward_time != nullptr) *forward_time = measured_forward;
    if (hamiltonian_time != nullptr) *hamiltonian_time = measured_hamiltonian;
    if (backward_time != nullptr) *backward_time = measured_backward;
}

template <typename T>
void copy_results(const HostResults<T>& host_results,
                  size_t parameter_count,
                  double* out_energy,
                  T* out_gradient) {
    *out_energy = host_results.energy.get()[0];
    for (size_t parameter = 0; parameter < parameter_count; ++parameter) {
        out_gradient[parameter] =
            static_cast<T>(host_results.gradients.get()[parameter]);
    }
}

template <typename T>
void run_typed(int circuit,
               int qubits,
               int layers,
               int steps,
               int warmup_steps,
               int device_id,
               const T* host_parameters,
               size_t parameter_count,
               const char* forward_phase_plan,
               const char* backward_phase_plan,
               double* out_energy,
               T* out_gradient,
               double* out_forward_times,
               double* out_hamiltonian_times,
               double* out_backward_times,
               SadMemoryInfo* out_memory) {
    const RunConfig config{
        circuit,
        qubits,
        layers,
        steps,
        warmup_steps,
        parameter_count,
        forward_phase_plan == nullptr ? "" : forward_phase_plan,
        backward_phase_plan == nullptr ? "" : backward_phase_plan};
    validate_inputs(circuit,
                    qubits,
                    layers,
                    steps,
                    warmup_steps,
                    parameter_count);
    const DeviceRunInfo device_info = prepare_device(device_id);
    PreparedWorkspace<T> workspace(config, device_info);
    prepare_workspace(&workspace, host_parameters, device_info, out_memory);
    HostResults<T> host_results(parameter_count);

    for (int warmup = 0; warmup < warmup_steps; ++warmup) {
        run_step(&workspace, &host_results, nullptr, nullptr, nullptr);
    }
    for (int step = 0; step < steps; ++step) {
        run_step(&workspace,
                 &host_results,
                 out_forward_times + step,
                 out_hamiltonian_times + step,
                 out_backward_times + step);
    }
    copy_results(host_results, parameter_count, out_energy, out_gradient);
}

}  // namespace sad
