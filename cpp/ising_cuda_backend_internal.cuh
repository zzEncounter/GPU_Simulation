#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/complex.h>

namespace standalone_backend {
namespace detail {

using Complex = thrust::complex<double>;

constexpr int THREADS = 256;

void check_cuda(cudaError_t status, const char *context);
void check_cublas(cublasStatus_t status, const char *context);
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
                                  bool inverse);
void launch_apply_ryrz(Complex *state, std::size_t size, std::size_t wire,
                       double theta_ry, double theta_rz);
void launch_apply_hamiltonian(Complex *out, const Complex *state,
                              std::size_t size, std::size_t num_qubits,
                              double field);

auto complex_inner_product(const Complex *lhs, const Complex *rhs,
                           std::size_t size) -> Complex;

auto fused_ryrz_gradients(const Complex *lambda, const Complex *state,
                          std::size_t state_size, std::size_t wire,
                          double theta_ry, double theta_rz)
    -> std::pair<double, double>;

void launch_fill_identity_matrices(Complex *mats, std::size_t batch,
                                   std::size_t dim);
void launch_prepare_downsweep_buffers(Complex *mats,
                                      const int *left_indices,
                                      const int *right_indices,
                                      Complex *left_tmp, std::size_t num_pairs,
                                      std::size_t mat_elements);
void launch_scatter_matrices(const Complex *source_mats,
                             const int *target_indices, Complex *target_mats,
                             std::size_t batch, std::size_t mat_elements);
void launch_gather_vectors(const Complex *source_vectors,
                           const int *source_indices, Complex *target_vectors,
                           std::size_t batch, std::size_t vector_size);
void launch_build_adjoint_batch(const Complex *source, Complex *target,
                                std::size_t batch, std::size_t dim);
void launch_reduce_real_inner_products(const Complex *lhs,
                                       const Complex *rhs, double *out,
                                       std::size_t batch,
                                       std::size_t vector_size,
                                       double scale);

} // namespace detail
} // namespace standalone_backend
