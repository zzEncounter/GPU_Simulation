#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace standalone_backend {

struct EnergyGradResult {
    double energy;
    std::vector<double> gradient;
    double forward_ms{0.0};
    double back_ms{0.0};
    double gradient_ms{0.0};
    double total_ms{0.0};
};

struct DenseScanExperimentResult {
    double energy;
    std::vector<double> gradient;
    std::vector<double> forward_states_ri;
    std::vector<double> backward_states_ri;
    double cpu_reference_ms;
    double gpu_scan_ms;
    double sequential_statevector_ms;
    std::size_t num_forward_states;
    std::size_t num_backward_states;
    std::size_t state_size;
};

class RingIsingCudaBackend {
  public:
    struct Impl;

    RingIsingCudaBackend(std::size_t num_qubits, std::size_t num_layers,
                         double field, const std::string &gradient_strategy,
                         bool fuse_ring_cnot_layer,
                         std::size_t checkpoint_interval_ops);
    ~RingIsingCudaBackend();

    RingIsingCudaBackend(const RingIsingCudaBackend &) = delete;
    auto operator=(const RingIsingCudaBackend &)
        -> RingIsingCudaBackend & = delete;
    RingIsingCudaBackend(RingIsingCudaBackend &&) noexcept;
    auto operator=(RingIsingCudaBackend &&) noexcept
        -> RingIsingCudaBackend &;

    auto energy_and_grad(const double *params, std::size_t num_params,
                         bool measure_timings = true,
                         bool compute_gradient = true)
        -> EnergyGradResult;
    auto dense_scan_experiment(const double *params, std::size_t num_params)
        -> DenseScanExperimentResult;

  private:
    std::unique_ptr<Impl> impl_;
};

EnergyGradResult energy_and_grad(std::size_t num_qubits,
                                 std::size_t num_layers, double field,
                                 const std::string &gradient_strategy,
                                 bool fuse_ring_cnot_layer,
                                 const double *params, std::size_t num_params,
                                 std::size_t checkpoint_interval_ops,
                                 bool measure_timings = true,
                                 bool compute_gradient = true);

DenseScanExperimentResult dense_scan_experiment(
    std::size_t num_qubits, std::size_t num_layers, double field,
    bool fuse_ring_cnot_layer, const double *params, std::size_t num_params);

} // namespace standalone_backend
