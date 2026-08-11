#pragma once

#include "../core/cuda_common.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace sad {

template <typename T>
struct DiagonalLookupData {
    std::vector<Complex<T>> factors;
    std::vector<size_t> offsets_by_parameter;
};

template <typename T>
struct InitialStateLookupData {
    std::vector<Complex<T>> factors;
};

constexpr size_t kNoParameterOffset = std::numeric_limits<size_t>::max();

template <typename T, NonDiagonalGate Gate>
auto build_initial_product_lookup(int qubits,
                                  const T* parameters,
                                  size_t rotation_offset,
                                  size_t rz_offset = kNoParameterOffset)
    -> InitialStateLookupData<T> {
    InitialStateLookupData<T> data;
    const int chunk_count =
        (qubits + kDiagonalLookupBits - 1) / kDiagonalLookupBits;
    data.factors.reserve(chunk_count * kDiagonalLookupSize);
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        for (int code = 0; code < kDiagonalLookupSize; ++code) {
            Complex<T> value{static_cast<T>(1), static_cast<T>(0)};
            for (int bit = 0; bit < kDiagonalLookupBits; ++bit) {
                const int qubit = chunk * kDiagonalLookupBits + bit;
                if (qubit >= qubits) {
                    break;
                }
                const bool one = ((code >> bit) & 1) != 0;
                const T half_angle =
                    parameters[rotation_offset + qubit] * static_cast<T>(0.5);
                const T sine = static_cast<T>(std::sin(half_angle));
                const T cosine = static_cast<T>(std::cos(half_angle));
                Complex<T> factor;
                if constexpr (Gate == NonDiagonalGate::RY) {
                    factor = {one ? sine : cosine, static_cast<T>(0)};
                } else {
                    factor = one ? Complex<T>{static_cast<T>(0), -sine}
                                 : Complex<T>{cosine, static_cast<T>(0)};
                }
                if (rz_offset != kNoParameterOffset) {
                    const T rz_half =
                        parameters[rz_offset + qubit] * static_cast<T>(0.5);
                    const T rz_sine = static_cast<T>(std::sin(rz_half));
                    const T rz_cosine = static_cast<T>(std::cos(rz_half));
                    const Complex<T> rz_factor{
                        rz_cosine, one ? rz_sine : -rz_sine};
                    factor = multiply(factor, rz_factor);
                }
                value = multiply(value, factor);
            }
            data.factors.push_back(value);
        }
    }
    return data;
}

template <typename T>
auto build_plus_rx_product_lookup(int qubits,
                                  const T* parameters,
                                  size_t rotation_offset)
    -> InitialStateLookupData<T> {
    InitialStateLookupData<T> data;
    const int chunk_count =
        (qubits + kDiagonalLookupBits - 1) / kDiagonalLookupBits;
    data.factors.reserve(chunk_count * kDiagonalLookupSize);
    const T inverse_sqrt_two =
        static_cast<T>(0.707106781186547524400844362104849039);
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        Complex<T> chunk_factor{static_cast<T>(1), static_cast<T>(0)};
        for (int bit = 0; bit < kDiagonalLookupBits; ++bit) {
            const int qubit = chunk * kDiagonalLookupBits + bit;
            if (qubit >= qubits) break;
            const T half_angle =
                parameters[rotation_offset + qubit] * static_cast<T>(0.5);
            chunk_factor = multiply(
                chunk_factor,
                Complex<T>{inverse_sqrt_two *
                               static_cast<T>(std::cos(half_angle)),
                           -inverse_sqrt_two *
                               static_cast<T>(std::sin(half_angle))});
        }
        for (int code = 0; code < kDiagonalLookupSize; ++code) {
            data.factors.push_back(chunk_factor);
        }
    }
    return data;
}

template <typename T>
void append_diagonal_lookup_group(const T* parameters,
                                  size_t parameter_offset,
                                  int gate_count,
                                  DiagonalLookupData<T>* data) {
    data->offsets_by_parameter[parameter_offset] = data->factors.size();
    const int chunk_count =
        (gate_count + kDiagonalLookupBits - 1) / kDiagonalLookupBits;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        for (int code = 0; code < kDiagonalLookupSize; ++code) {
            T phase = 0;
            for (int bit = 0; bit < kDiagonalLookupBits; ++bit) {
                const int gate = chunk * kDiagonalLookupBits + bit;
                if (gate < gate_count) {
                    const int eigenvalue = ((code >> bit) & 1) ? -1 : 1;
                    phase -= static_cast<T>(0.5) *
                             parameters[parameter_offset + gate] *
                             static_cast<T>(eigenvalue);
                }
            }
            data->factors.push_back(
                {static_cast<T>(std::cos(phase)), static_cast<T>(std::sin(phase))});
        }
    }
}

template <typename T>
void append_shared_diagonal_lookup_group(const T* parameters,
                                         size_t parameter_offset,
                                         int gate_count,
                                         DiagonalLookupData<T>* data) {
    data->offsets_by_parameter[parameter_offset] = data->factors.size();
    const int chunk_count =
        (gate_count + kDiagonalLookupBits - 1) / kDiagonalLookupBits;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        for (int code = 0; code < kDiagonalLookupSize; ++code) {
            int eigenvalue_sum = 0;
            for (int bit = 0; bit < kDiagonalLookupBits; ++bit) {
                const int gate = chunk * kDiagonalLookupBits + bit;
                if (gate < gate_count) {
                    eigenvalue_sum += ((code >> bit) & 1) ? -1 : 1;
                }
            }
            const T phase = -static_cast<T>(0.5) *
                            parameters[parameter_offset] *
                            static_cast<T>(eigenvalue_sum);
            data->factors.push_back(
                {static_cast<T>(std::cos(phase)),
                 static_cast<T>(std::sin(phase))});
        }
    }
}

inline void build_phase_maps(int qubits,
                             int tile_bits,
                             bool fixed_low_lanes,
                             std::vector<int>* selected_maps,
                             std::vector<int>* target_masks) {
    const int active_tile_bits = std::min(qubits, tile_bits);
    const int high_capacity = tile_bits - kLaneBits;
    const int remaining = std::max(0, qubits - tile_bits);
    const int phase_count = fixed_low_lanes && remaining > 0
                                ? 1 + (remaining + high_capacity - 1) /
                                          high_capacity
                                : (qubits + tile_bits - 1) / tile_bits;
    selected_maps->assign(phase_count * tile_bits, -1);
    target_masks->assign(phase_count, 0);
    for (int phase = 0; phase < phase_count; ++phase) {
        const int first = phase == 0
                              ? 0
                              : tile_bits + (phase - 1) * high_capacity;
        const int capacity = phase == 0 || !fixed_low_lanes
                                 ? tile_bits
                                 : high_capacity;
        const int compact_first = phase * tile_bits;
        const int target_first = fixed_low_lanes ? first : compact_first;
        const int count = std::min(capacity, qubits - target_first);
        int filled = 0;
        if (fixed_low_lanes && phase > 0) {
            for (int qubit = 0; qubit < std::min(kLaneBits, qubits); ++qubit) {
                (*selected_maps)[phase * tile_bits + filled++] = qubit;
            }
        }
        for (int target = 0; target < count; ++target) {
            const int slot = filled++;
            (*selected_maps)[phase * tile_bits + slot] = target_first + target;
            (*target_masks)[phase] |= 1 << slot;
        }
        for (int qubit = 0;
             filled < active_tile_bits && qubit < qubits;
             ++qubit) {
            bool used = false;
            for (int slot = 0; slot < filled; ++slot) {
                used |= (*selected_maps)[phase * tile_bits + slot] == qubit;
            }
            if (!used) {
                (*selected_maps)[phase * tile_bits + filled++] = qubit;
            }
        }
    }
}

inline void build_bond_phase_maps(int qubits,
                                  int tile_bits,
                                  int parity,
                                  std::vector<int>* selected_maps,
                                  std::vector<int>* pair_counts) {
    const int active_tile_bits = std::min(qubits, tile_bits);
    const int pairs_per_phase = tile_bits / 2;
    const int bond_count = qubits / 2;
    const int phase_count =
        (bond_count + pairs_per_phase - 1) / pairs_per_phase;
    selected_maps->assign(phase_count * tile_bits, -1);
    pair_counts->assign(phase_count, 0);
    for (int phase = 0; phase < phase_count; ++phase) {
        int filled = 0;
        const int first_bond = phase * pairs_per_phase;
        const int count = std::min(pairs_per_phase,
                                   bond_count - first_bond);
        (*pair_counts)[phase] = count;
        for (int bond = 0; bond < count; ++bond) {
            const int left = parity + 2 * (first_bond + bond);
            const int right = (left + 1) % qubits;
            (*selected_maps)[phase * tile_bits + filled++] = left;
            (*selected_maps)[phase * tile_bits + filled++] = right;
        }
        for (int qubit = 0;
             filled < active_tile_bits && qubit < qubits;
             ++qubit) {
            bool used = false;
            for (int slot = 0; slot < filled; ++slot) {
                used |= (*selected_maps)[phase * tile_bits + slot] == qubit;
            }
            if (!used) {
                (*selected_maps)[phase * tile_bits + filled++] = qubit;
            }
        }
    }
}


}  // namespace sad
