#ifndef SAD_API_H
#define SAD_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum SadCircuit {
    SAD_CIRCUIT_RA_HEA = 0,
    SAD_CIRCUIT_SU2_HEA = 1,
    SAD_CIRCUIT_RZZ_HEA = 2,
    SAD_CIRCUIT_QAOA = 3,
    SAD_CIRCUIT_XXZ_HVA = 4,
    SAD_CIRCUIT_MERA = 5,
    SAD_CIRCUIT_EQUIVARIANT_QNN = 6,
    SAD_CIRCUIT_DATA_REUPLOADING = 7,
    SAD_CIRCUIT_QAOA_NS = 8,
    SAD_CIRCUIT_QAOA_BD = 9,
};

enum SadPrecision {
    SAD_PRECISION_FLOAT32 = 0,
    SAD_PRECISION_FLOAT64 = 1,
};

typedef struct SadMemoryInfo {
    uint64_t state_vector_bytes;
    uint64_t total_workspace_bytes;
    uint64_t device_free_before_bytes;
    uint64_t device_free_after_alloc_bytes;
    uint64_t device_total_bytes;
} SadMemoryInfo;

/*
 * Run repeated energy + adjoint-gradient evaluations.
 *
 * params and out_grad use float or double according to precision. Timing arrays
 * have `steps` entries and are returned in seconds. Device allocation, H2D
 * parameter upload, and warmup calls are intentionally outside measured timing.
 * Each split is native wall-clock time with an event synchronization at the end.
 * State initialization is part of forward; energy/gradient D2H materialization
 * is part of Hamiltonian/backward respectively.
 */
int sad_energy_and_grad(
    int precision,
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
    size_t error_message_size);

const char* sad_version(void);

#ifdef __cplusplus
}
#endif

#endif
