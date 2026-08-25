#ifndef CUQUANTUM_API_H
#define CUQUANTUM_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum CuQuantumGateKind {
    CUQUANTUM_GATE_RX = 0,
    CUQUANTUM_GATE_RY = 1,
    CUQUANTUM_GATE_RZ = 2,
    CUQUANTUM_GATE_RZZ = 3,
    CUQUANTUM_GATE_CNOT = 4,
};

typedef struct CuQuantumGate {
    int kind;
    int wire0;
    int wire1;
    int parameter;
    double angle;
} CuQuantumGate;

int cuquantum_energy_and_grad(
    int circuit,
    int qubits,
    const CuQuantumGate* gates,
    size_t gate_count,
    size_t parameter_count,
    double* out_energy,
    double* out_gradient,
    char* error_message,
    size_t error_message_size);

const char* cuquantum_version(void);

#ifdef __cplusplus
}
#endif

#endif
