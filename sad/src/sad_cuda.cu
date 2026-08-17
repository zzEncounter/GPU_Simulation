#include "sad_api.h"
#include "runtime/runner.cuh"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace sad {

inline void write_error(char* destination,
                        size_t size,
                        const std::string& message) {
    if (destination == nullptr || size == 0) return;
    std::snprintf(destination, size, "%s", message.c_str());
}

}  // namespace sad


extern "C" int sad_energy_and_grad(int precision,
                                    int circuit,
                                    int qubits,
                                    int layers,
                                    int steps,
                                    int warmup_steps,
                                    int device,
                                    const void* params,
                                    size_t parameter_count,
                                    const char* forward_phase_plan,
                                    const char* backward_phase_plan,
                                    double* out_energy,
                                    void* out_grad,
                                    double* out_forward_times_s,
                                    double* out_hamiltonian_times_s,
                                    double* out_backward_times_s,
                                    SadMemoryInfo* out_memory,
                                    char* error_message,
                                    size_t error_message_size) {
    try {
        if (params == nullptr || out_energy == nullptr || out_grad == nullptr ||
            out_forward_times_s == nullptr || out_hamiltonian_times_s == nullptr ||
            out_backward_times_s == nullptr || out_memory == nullptr) {
            throw std::invalid_argument("null pointer passed to sad_energy_and_grad");
        }
        if (precision == SAD_PRECISION_FLOAT32) {
            sad::run_typed<float>(circuit,
                                  qubits,
                                  layers,
                                  steps,
                                  warmup_steps,
                                  device,
                                  static_cast<const float*>(params),
                                  parameter_count,
                                  forward_phase_plan,
                                  backward_phase_plan,
                                  out_energy,
                                  static_cast<float*>(out_grad),
                                  out_forward_times_s,
                                  out_hamiltonian_times_s,
                                  out_backward_times_s,
                                  out_memory);
        } else if (precision == SAD_PRECISION_FLOAT64) {
            sad::run_typed<double>(circuit,
                                   qubits,
                                   layers,
                                   steps,
                                   warmup_steps,
                                   device,
                                   static_cast<const double*>(params),
                                   parameter_count,
                                   forward_phase_plan,
                                   backward_phase_plan,
                                   out_energy,
                                   static_cast<double*>(out_grad),
                                   out_forward_times_s,
                                   out_hamiltonian_times_s,
                                   out_backward_times_s,
                                   out_memory);
        } else {
            throw std::invalid_argument("unknown precision id");
        }
        sad::write_error(error_message, error_message_size, "");
        return 0;
    } catch (const std::exception& exception) {
        sad::write_error(error_message, error_message_size, exception.what());
        return 1;
    } catch (...) {
        sad::write_error(error_message, error_message_size, "unknown C++ exception");
        return 2;
    }
}

extern "C" const char* sad_version(void) {
    return "0.1.0";
}
