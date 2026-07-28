#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <custatevec.h>
#include <thrust/complex.h>

namespace standalone_backend {

enum class OpKind : int { RY, RZ, CNOT, FusedRYRZ, RotationLayer, RingCNOTLayer };

struct OpDesc {
    OpKind kind;
    std::size_t wire0;
    std::size_t wire1;
    double theta0;
    double theta1;
    std::size_t param_index0;
    std::size_t param_index1;
    bool is_parametric;
};

namespace detail {

using Complex = thrust::complex<double>;

constexpr int THREADS = 256;

enum class RotationChunkKernelPreference : int {
    Cooperative,
    CooperativePair512,
    Register,
    RegisterDoubleBuffer,           // Phase 1: forward Register kernel + grid-stride double buffer
    CooperativeDoubleBuffer,        // Phase 2: forward Cooperative kernel + shared-mem double buffer
    CooperativeBackwardDoubleBuffer // Phase 3: backward Cooperative kernel + shared-mem double buffer
};

constexpr std::size_t PRODUCT_STATE_INIT_MAX_QUBITS = 32;

void check_cuda(cudaError_t status, const char *context);
void check_cublas(cublasStatus_t status, const char *context);
void check_custatevec(custatevecStatus_t status, const char *context);
void maybe_synchronize_cuda(const char *context);

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t size) { allocate(size); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    auto operator=(const DeviceBuffer &) -> DeviceBuffer & = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    auto operator=(DeviceBuffer &&other) noexcept -> DeviceBuffer & {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() { reset(); }

    void allocate(std::size_t size) {
        reset();
        if (size == 0) {
            return;
        }
        check_cuda(cudaMalloc(&ptr_, sizeof(T) * size), "cudaMalloc");
        size_ = size;
    }

    void reset() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    [[nodiscard]] auto get() const -> T * { return ptr_; }
    [[nodiscard]] auto size() const -> std::size_t { return size_; }

  private:
    T *ptr_{nullptr};
    std::size_t size_{0};
};

void launch_init_zero_state(Complex *state, std::size_t size);
void launch_apply_ry(Complex *state, std::size_t size, std::size_t wire,
                     double theta);
void launch_apply_dry(Complex *out, const Complex *state, std::size_t size,
                      std::size_t wire, double theta);
void launch_apply_rz(Complex *state, std::size_t size, std::size_t wire,
                     double theta);
void launch_apply_drz(Complex *out, const Complex *state, std::size_t size,
                      std::size_t wire, double theta);
void launch_apply_cnot(Complex *state, std::size_t size, std::size_t control,
                       std::size_t target);
void launch_apply_ring_cnot_layer(Complex *out, const Complex *in,
                                  std::size_t size, std::size_t num_qubits,
                                  bool inverse, cudaStream_t stream = 0);
void launch_apply_ryrz(Complex *state, std::size_t size, std::size_t wire,
                       double theta_ry, double theta_rz,
                       bool inverse = false, cudaStream_t stream = 0);
void launch_init_ryrz_product_state(Complex *state, std::size_t size,
                                    std::size_t num_qubits,
                                    const double *layer_params,
                                    cudaStream_t stream = 0);
void launch_apply_ryrz_rotation_chunk(Complex *state, std::size_t size,
                                      std::size_t chunk_start,
                                      std::size_t chunk_width,
                                      const double *theta_ry,
                                      const double *theta_rz,
                                      RotationChunkKernelPreference
                                          kernel_preference,
                                      cudaStream_t stream = 0);
void launch_inverse_walk_ry_step(Complex *current, Complex *lambda,
                                 std::size_t size, std::size_t wire,
                                 double theta, double *out_gradient);
void launch_inverse_walk_rz_step(Complex *current, Complex *lambda,
                                 std::size_t size, std::size_t wire,
                                 double theta, double *out_gradient);
void launch_inverse_walk_ry_gradient(const Complex *current,
                                     const Complex *lambda, std::size_t size,
                                     std::size_t wire, double theta,
                                     double *out_gradient);
void launch_inverse_walk_rz_gradient(const Complex *current,
                                     const Complex *lambda, std::size_t size,
                                     std::size_t wire, double theta,
                                     double *out_gradient);
void launch_inverse_walk_ryrz_step(Complex *current, Complex *lambda,
                                   std::size_t size, std::size_t wire,
                                   double theta_ry, double theta_rz,
                                   double *out_theta_gradient,
                                   double *out_phi_gradient);
void launch_inverse_walk_ryrz_rotation_chunk(
    Complex *current, Complex *lambda, std::size_t size,
    std::size_t chunk_start, std::size_t chunk_width,
    const double *theta_ry, const double *theta_rz, double *out_gradients);
// Phase 3: backward cooperative double-buffer launcher
void launch_inverse_walk_ryrz_rotation_chunk_db(
    Complex *current, Complex *lambda, std::size_t size,
    std::size_t chunk_start, std::size_t chunk_width,
    const double *theta_ry, const double *theta_rz, double *out_gradients);
void launch_inverse_ring_cnot_then_w4_rotation_chunk(
    const Complex *current_in, const Complex *lambda_in, Complex *current_out,
    Complex *lambda_out, std::size_t size, std::size_t num_qubits,
    std::size_t chunk_start, const double *theta_ry, const double *theta_rz,
    double *out_gradients);
void launch_inverse_walk_cnot_step(Complex *current, Complex *lambda,
                                   std::size_t size, std::size_t control,
                                   std::size_t target);
void launch_apply_hamiltonian(Complex *out, const Complex *state,
                              std::size_t size, std::size_t num_qubits,
                              double field);
auto hamiltonian_energy_partial_count(std::size_t size) -> std::size_t;
void launch_apply_hamiltonian_energy_partials(
    Complex *out, const Complex *state, Complex *energy_partials,
    std::size_t size, std::size_t num_qubits, double field);

auto complex_inner_product(const Complex *lhs, const Complex *rhs,
                           std::size_t size) -> Complex;
auto sum_real_parts(const Complex *values, std::size_t size) -> double;

void launch_fill_identity_matrices(Complex *mats, std::size_t batch,
                                   std::size_t dim);
void launch_scatter_parent_vectors(const Complex *parent_vectors,
                                   Complex *child_vectors,
                                   std::size_t parent_count,
                                   std::size_t vector_size);
void launch_fill_rotation_layer_matrices(Complex *gate_mats,
                                         const int *param_gate_indices,
                                         const double *params,
                                         std::size_t num_layers,
                                         std::size_t num_qubits,
                                         std::size_t dim);
void launch_fill_ring_cnot_layer_matrices(Complex *gate_mats,
                                          std::size_t num_layers,
                                          std::size_t num_qubits,
                                          std::size_t dim);
void launch_rotation_layer_dense_gradient_tail(
    const int *param_gate_indices, const double *params,
    const Complex *psi_before, const Complex *eta_before, double *out,
    std::size_t num_layers, std::size_t num_qubits, std::size_t vector_size);

void launch_simulate_blocks_forward(const OpDesc *ops, std::size_t num_blocks,
                                    std::size_t block_size, std::size_t num_ops,
                                    std::size_t num_qubits, std::size_t dim,
                                    const Complex *boundary_states,
                                    Complex *forward_states);
void launch_simulate_blocks_backward_and_gradient(
    const OpDesc *ops, std::size_t num_blocks, std::size_t block_size,
    std::size_t num_ops, std::size_t num_qubits, std::size_t dim,
    const Complex *lambda_boundaries, const Complex *forward_states,
    double *out_gradients);

} // namespace detail
} // namespace standalone_backend
