#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace standalone_backend {

struct EnergyGradResult {
    double energy{0.0};
    std::vector<double> gradient;
    std::vector<std::pair<std::string, double>> stage_timings_ms;
};

class RingIsingCudaBackend {
  public:
    struct Impl;

    RingIsingCudaBackend(std::size_t num_qubits, std::size_t num_layers,
                         double field, const std::string &gradient_strategy,
                         std::size_t mode2_rotation_chunk_width = 8);
    ~RingIsingCudaBackend();

    RingIsingCudaBackend(const RingIsingCudaBackend &) = delete;
    auto operator=(const RingIsingCudaBackend &)
        -> RingIsingCudaBackend & = delete;
    RingIsingCudaBackend(RingIsingCudaBackend &&) noexcept;
    auto operator=(RingIsingCudaBackend &&) noexcept
        -> RingIsingCudaBackend &;

    auto energy_and_grad(const double *params, std::size_t num_params,
                         bool compute_gradient = true,
                         bool profile = false) -> EnergyGradResult;

  private:
    std::unique_ptr<Impl> impl_;
};

EnergyGradResult energy_and_grad(std::size_t num_qubits,
                                 std::size_t num_layers, double field,
                                 const std::string &gradient_strategy,
                                 const double *params, std::size_t num_params,
                                 bool compute_gradient = true,
                                 bool profile = false,
                                 std::size_t mode2_rotation_chunk_width = 8);

} // namespace standalone_backend
