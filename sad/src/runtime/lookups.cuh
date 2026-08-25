#pragma once

#include "../core/cuda_common.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
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

struct PhaseTargetClasses {
    int lane = 0;
    int reg = 0;
    int warp = 0;
};

inline bool parse_phase_class_plan(
    const std::string& value,
    std::vector<PhaseTargetClasses>* phases) {
    std::stringstream stream(value);
    std::string phase;
    while (std::getline(stream, phase, '-')) {
        PhaseTargetClasses classes;
        char extra = 0;
        if (std::sscanf(phase.c_str(),
                        "L%dR%dW%d%c",
                        &classes.lane,
                        &classes.reg,
                        &classes.warp,
                        &extra) != 3 ||
            classes.lane < 0 || classes.reg < 0 || classes.warp < 0) {
            return false;
        }
        phases->push_back(classes);
    }
    return !phases->empty();
}

inline bool build_class_phase_maps(
    int qubits,
    int tile_bits,
    int register_bits,
    const std::string& family,
    const std::vector<PhaseTargetClasses>& phases,
    std::vector<int>* selected_maps,
    std::vector<int>* target_masks) {
    const int warp_bits = tile_bits - kLaneBits - register_bits;
    const bool fixed = family == "fixed" || family == "pairs";
    const bool pairs = family == "pairs";
    if (family != "compact" && !fixed) return false;
    if (pairs && warp_bits < 1) return false;

    int target_total = 0;
    for (const auto& phase : phases) {
        target_total += phase.lane + phase.reg + phase.warp;
    }
    if (target_total != qubits) return false;
    if (fixed && phases.front().lane + phases.front().reg +
                         phases.front().warp <
                     kLaneBits + (pairs ? 1 : 0)) {
        return false;
    }

    selected_maps->assign(phases.size() * tile_bits, -1);
    target_masks->assign(phases.size(), 0);
    int target = 0;
    for (size_t phase_index = 0; phase_index < phases.size(); ++phase_index) {
        const auto& phase = phases[phase_index];
        const bool reserve_low = fixed && phase_index > 0;
        const bool reserve_pair = pairs && phase_index > 0;
        if (phase.lane > kLaneBits || phase.reg > register_bits ||
            phase.warp > warp_bits || (reserve_low && phase.lane != 0) ||
            (reserve_pair && phase.warp > warp_bits - 1)) {
            return false;
        }
        const int base = static_cast<int>(phase_index) * tile_bits;
        if (reserve_low) {
            for (int lane = 0; lane < kLaneBits; ++lane) {
                (*selected_maps)[base + lane] = lane;
            }
        }
        const int first_warp_slot = kLaneBits + register_bits;
        if (reserve_pair) {
            (*selected_maps)[base + first_warp_slot] = kLaneBits;
        }

        const auto add_targets = [&](int first_slot, int count) {
            for (int offset = 0; offset < count; ++offset) {
                const int slot = first_slot + offset;
                (*selected_maps)[base + slot] = target++;
                (*target_masks)[phase_index] |= 1 << slot;
            }
        };
        add_targets(0, phase.lane);
        add_targets(kLaneBits, phase.reg);
        add_targets(first_warp_slot + (reserve_pair ? 1 : 0), phase.warp);

        for (int qubit = 0; qubit < qubits; ++qubit) {
            bool used = false;
            for (int slot = 0; slot < tile_bits; ++slot) {
                used |= (*selected_maps)[base + slot] == qubit;
            }
            if (used) continue;
            const auto empty = std::find(
                selected_maps->begin() + base,
                selected_maps->begin() + base + tile_bits,
                -1);
            if (empty == selected_maps->begin() + base + tile_bits) break;
            *empty = qubit;
        }
    }
    return true;
}

inline auto build_target_phase_owners(
    int qubits,
    int tile_bits,
    const std::vector<int>& selected_maps,
    const std::vector<int>& target_masks) -> std::vector<int> {
    if (selected_maps.size() != target_masks.size() * tile_bits) {
        throw std::invalid_argument("phase map and mask sizes do not match");
    }
    std::vector<int> owners(qubits, -1);
    for (size_t phase = 0; phase < target_masks.size(); ++phase) {
        for (int slot = 0; slot < tile_bits; ++slot) {
            if ((target_masks[phase] & (1 << slot)) == 0) continue;
            const int qubit = selected_maps[phase * tile_bits + slot];
            if (qubit < 0 || qubit >= qubits || owners[qubit] != -1) {
                throw std::invalid_argument(
                    "phase targets must cover every qubit exactly once");
            }
            owners[qubit] = static_cast<int>(phase);
        }
    }
    if (std::find(owners.begin(), owners.end(), -1) != owners.end()) {
        throw std::invalid_argument(
            "phase targets must cover every qubit exactly once");
    }
    return owners;
}

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

template <typename T>
void append_ring_rzz_compact_lookup_group(const T* parameters,
                                          size_t parameter_offset,
                                          int qubits,
                                          DiagonalLookupData<T>* data) {
    data->offsets_by_parameter[parameter_offset] = data->factors.size();
    // A closed binary ring has an even number of transitions, so only
    // q/2+1 distinct generator eigenvalues are reachable.
    for (int wall_pairs = 0; wall_pairs <= qubits / 2; ++wall_pairs) {
        const int domain_walls = 2 * wall_pairs;
        const int eigenvalue_sum = qubits - 2 * domain_walls;
        const T phase = -static_cast<T>(0.5) *
                        parameters[parameter_offset] *
                        static_cast<T>(eigenvalue_sum);
        data->factors.push_back(
            {static_cast<T>(std::cos(phase)),
             static_cast<T>(std::sin(phase))});
    }
}

// Non-shared ring RZZ uses only the two eigenvalue cases for each edge.
// Keeping this compact layout local to qaoa_ns avoids changing shared-QAOA
// lookup offsets and reduces per-edge storage from kDiagonalLookupSize values
// to two values.
template <typename T>
void append_nonshared_ring_rzz_lookup_group(const T* parameters,
                                            size_t parameter_offset,
                                            DiagonalLookupData<T>* data) {
    data->offsets_by_parameter[parameter_offset] = data->factors.size();
    const T angle = parameters[parameter_offset];
    const T half_angle = static_cast<T>(0.5) * angle;
    data->factors.push_back(
        {static_cast<T>(std::cos(-half_angle)),
         static_cast<T>(std::sin(-half_angle))});
    data->factors.push_back(
        {static_cast<T>(std::cos(half_angle)),
         static_cast<T>(std::sin(half_angle))});
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

// Preserve the original all-even -> all-odd dependency order while retaining
// each contiguous block in one tile: internal odd bonds follow that block's
// even bonds, and all cross-block odd bonds are emitted in a final phase.
inline void build_xxz_cross_matching_maps(
    int qubits,
    int tile_bits,
    std::vector<int>* selected_maps,
    std::vector<int>* bond_offsets,
    std::vector<int>* slot_pairs,
    std::vector<int>* edges) {
    const int active_tile_bits = std::min(qubits, tile_bits);
    const int block_width = active_tile_bits & ~1;
    std::vector<std::pair<int, int>> blocks;
    selected_maps->clear();
    bond_offsets->assign(1, 0);
    slot_pairs->clear();
    edges->clear();

    const auto append_selected = [&](const std::vector<int>& phase_qubits) {
        const int base = static_cast<int>(selected_maps->size());
        selected_maps->resize(base + tile_bits, -1);
        int filled = 0;
        for (const int qubit : phase_qubits) {
            (*selected_maps)[base + filled++] = qubit;
        }
        for (int qubit = 0;
             filled < active_tile_bits && qubit < qubits;
             ++qubit) {
            if (std::find(phase_qubits.begin(), phase_qubits.end(), qubit) ==
                phase_qubits.end()) {
                (*selected_maps)[base + filled++] = qubit;
            }
        }
    };

    for (int first = 0; first < qubits; first += block_width) {
        const int width = std::min(block_width, qubits - first);
        blocks.emplace_back(first, width);
        std::vector<int> phase_qubits;
        for (int offset = 0; offset < width; ++offset) {
            phase_qubits.push_back(first + offset);
        }
        append_selected(phase_qubits);
        const auto append_bond = [&](int first_slot, int edge) {
            slot_pairs->push_back(first_slot);
            slot_pairs->push_back(first_slot + 1);
            edges->push_back(edge);
        };
        for (int offset = 0; offset + 1 < width; offset += 2) {
            append_bond(offset, first + offset);
        }
        for (int offset = 1; offset + 1 < width; offset += 2) {
            append_bond(offset, first + offset);
        }
        bond_offsets->push_back(static_cast<int>(edges->size()));
    }

    std::vector<int> boundary_qubits;
    std::vector<std::pair<int, int>> boundary_bonds;
    for (size_t block = 0; block < blocks.size(); ++block) {
        const int left = blocks[block].first + blocks[block].second - 1;
        const int right = blocks[(block + 1) % blocks.size()].first;
        boundary_bonds.emplace_back(left, right);
        boundary_qubits.push_back(left);
        boundary_qubits.push_back(right);
    }
    append_selected(boundary_qubits);
    for (const auto [left, right] : boundary_bonds) {
        const int left_slot = static_cast<int>(
            std::find(boundary_qubits.begin(), boundary_qubits.end(), left) -
            boundary_qubits.begin());
        const int right_slot = static_cast<int>(
            std::find(boundary_qubits.begin(), boundary_qubits.end(), right) -
            boundary_qubits.begin());
        slot_pairs->push_back(left_slot);
        slot_pairs->push_back(right_slot);
        edges->push_back(left);
    }
    bond_offsets->push_back(static_cast<int>(edges->size()));
}


}  // namespace sad
