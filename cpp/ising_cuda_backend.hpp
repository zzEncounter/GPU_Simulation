#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace standalone_backend {

struct EnergyGradResult {
    double energy;
    std::vector<double> gradient;
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

    auto forward_energy(const double *params, std::size_t num_params) -> double;
    auto energy_and_grad(const double *params, std::size_t num_params)
        -> EnergyGradResult;

  private:
    std::unique_ptr<Impl> impl_;
};

double forward_energy(std::size_t num_qubits, std::size_t num_layers,
                      double field, const double *params,
                      std::size_t num_params,
                      bool fuse_ring_cnot_layer);

EnergyGradResult energy_and_grad(std::size_t num_qubits,
                                 std::size_t num_layers, double field,
                                 const std::string &gradient_strategy,
                                 bool fuse_ring_cnot_layer,
                                 const double *params, std::size_t num_params,
                                 std::size_t checkpoint_interval_ops);

} // namespace standalone_backend
