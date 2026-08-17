#pragma once

#include "sad_api.h"

#include "../core/cuda_common.cuh"
#include "../core/cuda_resources.cuh"
#include "circuit_execution.cuh"
#include "lookups.cuh"
#include "options.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sad {

struct RunConfig {
    int circuit;
    int qubits;
    int layers;
    int steps;
    int warmup_steps;
    size_t parameter_count;
    std::string forward_phase_plan;
    std::string backward_phase_plan;
};

struct DeviceRunInfo {
    cudaDeviceProp properties{};
    size_t free_before = 0;
    size_t total_memory = 0;
};

inline bool build_configured_phase_maps(
    const std::string& plan,
    const char* label,
    int qubits,
    int tile_bits,
    int register_bits,
    std::vector<int>* selected_maps,
    std::vector<int>* target_masks) {
    if (plan.empty()) return false;
    const size_t separator = plan.find(':');
    if (separator == std::string::npos) {
        throw std::invalid_argument(
            std::string(label) +
            " must be family:LxRyWz-LxRyWz-...");
    }
    const std::string family = plan.substr(0, separator);
    std::vector<PhaseTargetClasses> phases;
    if (!parse_phase_class_plan(plan.substr(separator + 1), &phases) ||
        !build_class_phase_maps(qubits,
                                tile_bits,
                                register_bits,
                                family,
                                phases,
                                selected_maps,
                                target_masks)) {
        throw std::invalid_argument(
            std::string("invalid ") + label + " for selected tile shape");
    }
    return true;
}

inline auto prepare_device(int device_id) -> DeviceRunInfo {
    SAD_CUDA_CHECK(cudaSetDevice(device_id));
    DeviceRunInfo info;
    SAD_CUDA_CHECK(cudaGetDeviceProperties(&info.properties, device_id));
    if constexpr (kRotationPersistent || kXxzPersistent || kRealPersistent ||
                  kPhasedRyPersistent) {
        if (!info.properties.cooperativeLaunch) {
            throw std::runtime_error(
                "selected CUDA device does not support cooperative launch");
        }
    }
    SAD_CUDA_CHECK(cudaMemGetInfo(&info.free_before, &info.total_memory));
    return info;
}

template <typename T>
struct PreparedWorkspace {
    PreparedWorkspace(const RunConfig& run_config,
                      const DeviceRunInfo& device_info)
        : config(run_config),
          state_size(1ull << config.qubits),
          execution_mode(read_execution_mode()),
          multiprocessors(device_info.properties.multiProcessorCount),
          ordinary_grid(ordinary_grid_size(state_size, multiprocessors)),
          rotation_coefficients(config.parameter_count),
          gradients(config.parameter_count),
          energy(1) {
        const uint64_t stored_elements =
            use_real_amplitude_state(config.circuit, execution_mode)
                ? state_size / 2
                : state_size;
        phi_a.allocate(stored_elements);
        phi_b.allocate(stored_elements);
        lambda_a.allocate(stored_elements);
        lambda_b.allocate(stored_elements);
    }

    RunConfig config;
    uint64_t state_size;
    ExecutionMode execution_mode;
    int multiprocessors;
    int ordinary_grid;
    int forward_phase_count = 0;
    int backward_phase_count = 0;
    int forward_xxz_even_phase_count = 0;
    int forward_xxz_odd_phase_count = 0;
    int backward_xxz_even_phase_count = 0;
    int backward_xxz_odd_phase_count = 0;

    DeviceBuffer<Complex<T>> phi_a;
    DeviceBuffer<Complex<T>> phi_b;
    DeviceBuffer<Complex<T>> lambda_a;
    DeviceBuffer<Complex<T>> lambda_b;
    DeviceBuffer<RotationCoefficients<T>> rotation_coefficients;
    DeviceBuffer<Complex<T>> initial_state_lookup;
    DeviceBuffer<Complex<T>> diagonal_lookup;
    DeviceBuffer<double> gradients;
    DeviceBuffer<double> energy;
    DeviceBuffer<int> forward_selected_maps;
    DeviceBuffer<int> forward_target_masks;
    DeviceBuffer<int> backward_selected_maps;
    DeviceBuffer<int> backward_target_masks;
    DeviceBuffer<int> backward_target_phases;
    DeviceBuffer<int> forward_xxz_even_selected_maps;
    DeviceBuffer<int> forward_xxz_even_pair_counts;
    DeviceBuffer<int> forward_xxz_odd_selected_maps;
    DeviceBuffer<int> forward_xxz_odd_pair_counts;
    DeviceBuffer<int> backward_xxz_even_selected_maps;
    DeviceBuffer<int> backward_xxz_even_pair_counts;
    DeviceBuffer<int> backward_xxz_odd_selected_maps;
    DeviceBuffer<int> backward_xxz_odd_pair_counts;
    DiagonalLookupData<T> diagonal_lookup_data;
};

template <typename T>
struct HostResults {
    explicit HostResults(size_t parameter_count)
        : energy(1), gradients(parameter_count) {}

    PinnedBuffer<double> energy;
    PinnedBuffer<double> gradients;
    EventPair timer;
};

template <typename T>
void prepare_workspace(PreparedWorkspace<T>* workspace,
                       const T* host_parameters,
                       const DeviceRunInfo& device_info,
                       SadMemoryInfo* out_memory) {
    const auto& config = workspace->config;
    workspace->diagonal_lookup_data = build_diagonal_lookups(
        config.circuit,
        config.qubits,
        config.layers,
        host_parameters,
        config.parameter_count);
    const auto initial_state_lookup_data = build_initial_state_lookup(
        config.circuit, config.qubits, host_parameters);

    workspace->initial_state_lookup.allocate(
        initial_state_lookup_data.factors.size());
    if (!workspace->diagonal_lookup_data.factors.empty()) {
        workspace->diagonal_lookup.allocate(
            workspace->diagonal_lookup_data.factors.size());
    }

    std::vector<RotationCoefficients<T>> host_rotation_coefficients(
        config.parameter_count);
    for (size_t parameter = 0; parameter < config.parameter_count; ++parameter) {
        const T half_angle = host_parameters[parameter] * static_cast<T>(0.5);
        host_rotation_coefficients[parameter] = {
            static_cast<T>(std::sin(half_angle)),
            static_cast<T>(std::cos(half_angle))};
    }

    std::vector<int> forward_selected_maps;
    std::vector<int> forward_target_masks;
    std::vector<int> backward_selected_maps;
    std::vector<int> backward_target_masks;
    std::vector<int> forward_xxz_even_selected_maps;
    std::vector<int> forward_xxz_even_pair_counts;
    std::vector<int> forward_xxz_odd_selected_maps;
    std::vector<int> forward_xxz_odd_pair_counts;
    std::vector<int> backward_xxz_even_selected_maps;
    std::vector<int> backward_xxz_even_pair_counts;
    std::vector<int> backward_xxz_odd_selected_maps;
    std::vector<int> backward_xxz_odd_pair_counts;
    if (!build_configured_phase_maps(config.forward_phase_plan,
                                      "forward phase plan",
                                      config.qubits,
                                      kForwardTileBits,
                                      kForwardRegisterBits,
                                      &forward_selected_maps,
                                      &forward_target_masks)) {
        build_phase_maps(config.qubits,
                         kForwardTileBits,
                         kForwardFixedLowLanes,
                         &forward_selected_maps,
                         &forward_target_masks);
    }
    if (!build_configured_phase_maps(config.backward_phase_plan,
                                      "backward phase plan",
                                      config.qubits,
                                      kTileBits,
                                      kRegisterBits,
                                      &backward_selected_maps,
                                      &backward_target_masks)) {
        build_phase_maps(config.qubits,
                         kTileBits,
                         kFixedLowLanes ||
                             ((config.circuit == SAD_CIRCUIT_RA_HEA ||
                               config.circuit == SAD_CIRCUIT_SU2_HEA) &&
                              config.qubits >= 18),
                         &backward_selected_maps,
                         &backward_target_masks);
    }
    if (config.circuit == SAD_CIRCUIT_XXZ_HVA) {
        if constexpr (kXxzCrossMatching) {
            build_xxz_cross_matching_maps(
                config.qubits,
                kForwardTileBits,
                &forward_xxz_even_selected_maps,
                &forward_xxz_even_pair_counts,
                &forward_xxz_odd_selected_maps,
                &forward_xxz_odd_pair_counts);
            build_xxz_cross_matching_maps(
                config.qubits,
                kTileBits,
                &backward_xxz_even_selected_maps,
                &backward_xxz_even_pair_counts,
                &backward_xxz_odd_selected_maps,
                &backward_xxz_odd_pair_counts);
        } else {
            build_bond_phase_maps(config.qubits,
                                  kForwardTileBits,
                                  0,
                                  &forward_xxz_even_selected_maps,
                                  &forward_xxz_even_pair_counts);
            build_bond_phase_maps(config.qubits,
                                  kForwardTileBits,
                                  1,
                                  &forward_xxz_odd_selected_maps,
                                  &forward_xxz_odd_pair_counts);
            build_bond_phase_maps(config.qubits,
                                  kTileBits,
                                  0,
                                  &backward_xxz_even_selected_maps,
                                  &backward_xxz_even_pair_counts);
            build_bond_phase_maps(config.qubits,
                                  kTileBits,
                                  1,
                                  &backward_xxz_odd_selected_maps,
                                  &backward_xxz_odd_pair_counts);
        }
    }
    workspace->forward_phase_count =
        static_cast<int>(forward_target_masks.size());
    workspace->backward_phase_count =
        static_cast<int>(backward_target_masks.size());
    const std::vector<int> backward_target_phases =
        build_target_phase_owners(config.qubits,
                                  kTileBits,
                                  backward_selected_maps,
                                  backward_target_masks);
    if constexpr (kXxzCrossMatching) {
        workspace->forward_xxz_even_phase_count =
            std::max(0,
                     static_cast<int>(forward_xxz_even_pair_counts.size()) - 1);
        workspace->forward_xxz_odd_phase_count = 0;
        workspace->backward_xxz_even_phase_count =
            std::max(0,
                     static_cast<int>(backward_xxz_even_pair_counts.size()) - 1);
        workspace->backward_xxz_odd_phase_count = 0;
    } else {
        workspace->forward_xxz_even_phase_count =
            static_cast<int>(forward_xxz_even_pair_counts.size());
        workspace->forward_xxz_odd_phase_count =
            static_cast<int>(forward_xxz_odd_pair_counts.size());
        workspace->backward_xxz_even_phase_count =
            static_cast<int>(backward_xxz_even_pair_counts.size());
        workspace->backward_xxz_odd_phase_count =
            static_cast<int>(backward_xxz_odd_pair_counts.size());
    }
    workspace->forward_selected_maps.allocate(forward_selected_maps.size());
    workspace->forward_target_masks.allocate(forward_target_masks.size());
    workspace->backward_selected_maps.allocate(backward_selected_maps.size());
    workspace->backward_target_masks.allocate(backward_target_masks.size());
    workspace->backward_target_phases.allocate(backward_target_phases.size());
    workspace->forward_xxz_even_selected_maps.allocate(
        forward_xxz_even_selected_maps.size());
    workspace->forward_xxz_even_pair_counts.allocate(
        forward_xxz_even_pair_counts.size());
    workspace->forward_xxz_odd_selected_maps.allocate(
        forward_xxz_odd_selected_maps.size());
    workspace->forward_xxz_odd_pair_counts.allocate(
        forward_xxz_odd_pair_counts.size());
    workspace->backward_xxz_even_selected_maps.allocate(
        backward_xxz_even_selected_maps.size());
    workspace->backward_xxz_even_pair_counts.allocate(
        backward_xxz_even_pair_counts.size());
    workspace->backward_xxz_odd_selected_maps.allocate(
        backward_xxz_odd_selected_maps.size());
    workspace->backward_xxz_odd_pair_counts.allocate(
        backward_xxz_odd_pair_counts.size());

    SAD_CUDA_CHECK(cudaMemcpy(workspace->rotation_coefficients.get(),
                              host_rotation_coefficients.data(),
                              workspace->rotation_coefficients.bytes(),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(workspace->initial_state_lookup.get(),
                              initial_state_lookup_data.factors.data(),
                              workspace->initial_state_lookup.bytes(),
                              cudaMemcpyHostToDevice));
    if (!workspace->diagonal_lookup_data.factors.empty()) {
        SAD_CUDA_CHECK(cudaMemcpy(
            workspace->diagonal_lookup.get(),
            workspace->diagonal_lookup_data.factors.data(),
            workspace->diagonal_lookup.bytes(),
            cudaMemcpyHostToDevice));
    }
    SAD_CUDA_CHECK(cudaMemcpy(workspace->forward_selected_maps.get(),
                              forward_selected_maps.data(),
                              workspace->forward_selected_maps.bytes(),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(workspace->forward_target_masks.get(),
                              forward_target_masks.data(),
                              workspace->forward_target_masks.bytes(),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(workspace->backward_selected_maps.get(),
                              backward_selected_maps.data(),
                              workspace->backward_selected_maps.bytes(),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(workspace->backward_target_masks.get(),
                              backward_target_masks.data(),
                              workspace->backward_target_masks.bytes(),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(workspace->backward_target_phases.get(),
                              backward_target_phases.data(),
                              workspace->backward_target_phases.bytes(),
                              cudaMemcpyHostToDevice));
    const auto copy_ints = [](DeviceBuffer<int>* destination,
                              const std::vector<int>& source) {
        if (!source.empty()) {
            SAD_CUDA_CHECK(cudaMemcpy(destination->get(),
                                      source.data(),
                                      destination->bytes(),
                                      cudaMemcpyHostToDevice));
        }
    };
    copy_ints(&workspace->forward_xxz_even_selected_maps,
              forward_xxz_even_selected_maps);
    copy_ints(&workspace->forward_xxz_even_pair_counts,
              forward_xxz_even_pair_counts);
    copy_ints(&workspace->forward_xxz_odd_selected_maps,
              forward_xxz_odd_selected_maps);
    copy_ints(&workspace->forward_xxz_odd_pair_counts,
              forward_xxz_odd_pair_counts);
    copy_ints(&workspace->backward_xxz_even_selected_maps,
              backward_xxz_even_selected_maps);
    copy_ints(&workspace->backward_xxz_even_pair_counts,
              backward_xxz_even_pair_counts);
    copy_ints(&workspace->backward_xxz_odd_selected_maps,
              backward_xxz_odd_selected_maps);
    copy_ints(&workspace->backward_xxz_odd_pair_counts,
              backward_xxz_odd_pair_counts);

    size_t free_after_alloc = 0;
    size_t ignored_total = 0;
    SAD_CUDA_CHECK(cudaMemGetInfo(&free_after_alloc, &ignored_total));
    out_memory->state_vector_bytes = workspace->phi_a.bytes();
    out_memory->total_workspace_bytes =
        4 * out_memory->state_vector_bytes +
        workspace->rotation_coefficients.bytes() +
        workspace->initial_state_lookup.bytes() +
        workspace->diagonal_lookup.bytes() + workspace->gradients.bytes() +
        workspace->energy.bytes() + workspace->forward_selected_maps.bytes() +
        workspace->forward_target_masks.bytes() +
        workspace->backward_selected_maps.bytes() +
        workspace->backward_target_masks.bytes() +
        workspace->forward_xxz_even_selected_maps.bytes() +
        workspace->forward_xxz_even_pair_counts.bytes() +
        workspace->forward_xxz_odd_selected_maps.bytes() +
        workspace->forward_xxz_odd_pair_counts.bytes() +
        workspace->backward_xxz_even_selected_maps.bytes() +
        workspace->backward_xxz_even_pair_counts.bytes() +
        workspace->backward_xxz_odd_selected_maps.bytes() +
        workspace->backward_xxz_odd_pair_counts.bytes();
    out_memory->device_free_before_bytes = device_info.free_before;
    out_memory->device_free_after_alloc_bytes = free_after_alloc;
    out_memory->device_total_bytes = device_info.total_memory;
}

}  // namespace sad
