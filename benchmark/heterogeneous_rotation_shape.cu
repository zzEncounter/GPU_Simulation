#ifndef SAD_SHAPE_NAMESPACE
#error "SAD_SHAPE_NAMESPACE must name the private shape namespace"
#endif
#ifndef SAD_SHAPE_WRAPPER
#error "SAD_SHAPE_WRAPPER must name the exported wrapper"
#endif

// Each object is compiled with a different private namespace.  This keeps the
// existing macro-specialized kernels isolated while allowing one executable to
// link several launch geometries and select one for every phase.
#define sad SAD_SHAPE_NAMESPACE
#include "kernels/diagonal.cuh"
#include "kernels/rotation.cuh"
#undef sad

extern "C" void SAD_SHAPE_WRAPPER(int direction,
                                  int gate,
                                  void* phi,
                                  void* lambda,
                                  const void* coefficients,
                                  double* gradients,
                                  int qubits,
                                  const int* selected,
                                  const int* target_mask,
                                  int multiprocessors) {
    using Complex = SAD_SHAPE_NAMESPACE::Complex<double>;
    using Coefficients = SAD_SHAPE_NAMESPACE::RotationCoefficients<double>;
    auto* typed_phi = static_cast<Complex*>(phi);
    auto* typed_lambda = static_cast<Complex*>(lambda);
    const auto* typed_coefficients = static_cast<const Coefficients*>(coefficients);
    if (direction == 0) {
        if (gate == 0) {
            SAD_SHAPE_NAMESPACE::launch_non_diagonal_forward<
                double, SAD_SHAPE_NAMESPACE::NonDiagonalGate::RX>(
                typed_phi,
                typed_coefficients,
                qubits,
                0,
                selected,
                target_mask,
                1,
                multiprocessors);
        } else {
            SAD_SHAPE_NAMESPACE::launch_non_diagonal_forward<
                double, SAD_SHAPE_NAMESPACE::NonDiagonalGate::RY>(
                typed_phi,
                typed_coefficients,
                qubits,
                0,
                selected,
                target_mask,
                1,
                multiprocessors);
        }
    } else if (gate == 0) {
        SAD_SHAPE_NAMESPACE::launch_non_diagonal_backward<
            double, SAD_SHAPE_NAMESPACE::NonDiagonalGate::RX>(
            typed_phi,
            typed_lambda,
            typed_coefficients,
            gradients,
            qubits,
            0,
            selected,
            target_mask,
            1,
            multiprocessors);
    } else {
        SAD_SHAPE_NAMESPACE::launch_non_diagonal_backward<
            double, SAD_SHAPE_NAMESPACE::NonDiagonalGate::RY>(
            typed_phi,
            typed_lambda,
            typed_coefficients,
            gradients,
            qubits,
            0,
            selected,
            target_mask,
            1,
            multiprocessors);
    }
}
