#include "kernels/diagonal.cuh"
#include "kernels/rotation.cuh"
#include "runtime/lookups.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace sad;

#ifndef SAD_MICRO_ITERATIONS
#define SAD_MICRO_ITERATIONS 50
#endif

namespace {

bool parse_counts(const std::string& value, std::vector<int>* counts) {
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        try {
            counts->push_back(std::stoi(token));
        } catch (...) {
            return false;
        }
    }
    return !counts->empty();
}

std::string join_ints(const std::vector<int>& values, char separator) {
    std::ostringstream stream;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) stream << separator;
        stream << values[index];
    }
    return stream.str();
}

std::string csv_safe(std::string value) {
    std::replace(value.begin(), value.end(), ',', ';');
    return value;
}

}  // namespace

bool build_pair_contiguous_phase_maps(int qubits,
                                      int tile_bits,
                                      int register_bits,
                                      std::vector<int>* selected,
                                      std::vector<int>* target_masks) {
    const int warp_bits = tile_bits - kLaneBits - register_bits;
    if (warp_bits < 1 || qubits < kLaneBits + 1) return false;
    const int high_capacity = tile_bits - (kLaneBits + 1);
    if (high_capacity < 1) return false;
    const int remaining = std::max(0, qubits - tile_bits);
    const int phase_count =
        1 + (remaining + high_capacity - 1) / high_capacity;
    selected->assign(phase_count * tile_bits, -1);
    target_masks->assign(phase_count, 0);

    const int first_count = std::min(qubits, tile_bits);
    for (int slot = 0; slot < first_count; ++slot) {
        (*selected)[slot] = slot;
        (*target_masks)[0] |= 1 << slot;
    }
    for (int phase = 1; phase < phase_count; ++phase) {
        const int base = phase * tile_bits;
        for (int lane = 0; lane < kLaneBits; ++lane) {
            (*selected)[base + lane] = lane;
        }
        const int first_warp_slot = kLaneBits + register_bits;
        (*selected)[base + first_warp_slot] = kLaneBits;
        int target = tile_bits + (phase - 1) * high_capacity;
        for (int slot = kLaneBits;
             slot < kLaneBits + register_bits && target < qubits;
             ++slot, ++target) {
            (*selected)[base + slot] = target;
            (*target_masks)[phase] |= 1 << slot;
        }
        for (int slot = first_warp_slot + 1;
             slot < tile_bits && target < qubits;
             ++slot, ++target) {
            (*selected)[base + slot] = target;
            (*target_masks)[phase] |= 1 << slot;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5 || argc > 7) {
        std::fprintf(stderr,
                     "usage: %s QUBITS rx|ry low|high|range:FIRST|fixed|"
                     "fixed:FIRST|pairs:FIRST|sequence:N,N,...|"
                     "compact-sequence:N,N,...|"
                     "fixed-sequence:N,N,...|full|full-fixed|full-pairs|"
                     "plan:compact|fixed|pairs:L5R2W1-L0R2W0... "
                     "forward|backward [all|none|lane|register|warp|COUNT] "
                     "[adaptive]\n",
                     argv[0]);
        return 2;
    }
    const int qubits = std::stoi(argv[1]);
    const std::string gate = argv[2];
    const std::string layout = argv[3];
    const std::string direction = argv[4];
    const std::string target_spec = argc == 6 ? argv[5] : "all";
    const bool adaptive_mailbox = argc == 7 && std::string(argv[6]) == "adaptive";
    if (argc == 7 && !adaptive_mailbox) return 2;
    const int tile_bits =
        direction == "forward" ? kForwardTileBits : kTileBits;
    const int register_bits = direction == "forward" ? kForwardRegisterBits
                                                       : kRegisterBits;
    const bool fixed_layout =
        layout == "fixed" || layout.rfind("fixed:", 0) == 0;
    const bool pair_layout =
        layout == "pairs" || layout.rfind("pairs:", 0) == 0;
    if (gate != "rx" && gate != "ry") return 2;

    std::vector<int> selected;
    std::vector<int> target_masks;
    if (layout == "full" || layout == "full-fixed") {
        build_phase_maps(
            qubits,
            tile_bits,
            layout == "full-fixed",
            &selected,
            &target_masks);
    } else if (layout == "full-pairs") {
        if (!build_pair_contiguous_phase_maps(
                qubits, tile_bits, register_bits, &selected, &target_masks)) {
            return 2;
        }
    } else if (layout.rfind("plan:", 0) == 0) {
        const size_t family_end = layout.find(':', 5);
        if (family_end == std::string::npos) return 2;
        const std::string family = layout.substr(5, family_end - 5);
        std::vector<PhaseTargetClasses> phases;
        if (!parse_phase_class_plan(layout.substr(family_end + 1), &phases) ||
            !build_class_phase_maps(qubits,
                                    tile_bits,
                                    register_bits,
                                    family,
                                    phases,
                                    &selected,
                                    &target_masks)) {
            return 2;
        }
    } else if (layout.rfind("sequence:", 0) == 0 ||
               layout.rfind("fixed-sequence:", 0) == 0 ||
               layout.rfind("compact-sequence:", 0) == 0) {
        const bool compact_sequence =
            layout.rfind("compact-sequence:", 0) == 0;
        const size_t prefix = compact_sequence
                                  ? std::string("compact-sequence:").size()
                              : layout.rfind("fixed-sequence:", 0) == 0
                                  ? std::string("fixed-sequence:").size()
                                  : std::string("sequence:").size();
        std::vector<int> counts;
        if (!parse_counts(layout.substr(prefix), &counts)) return 2;
        const int total = std::accumulate(counts.begin(), counts.end(), 0);
        if (counts.empty() || total != qubits) return 2;
        selected.assign(counts.size() * tile_bits, -1);
        target_masks.assign(counts.size(), 0);
        int target = 0;
        for (size_t phase = 0; phase < counts.size(); ++phase) {
            const int first_slot = phase == 0 || compact_sequence
                                       ? 0
                                       : kLaneBits;
            const int capacity = phase == 0 || compact_sequence
                                     ? tile_bits
                                     : tile_bits - kLaneBits;
            if (counts[phase] <= 0 || counts[phase] > capacity) return 2;
            if (!compact_sequence && phase > 0) {
                for (int slot = 0; slot < kLaneBits; ++slot) {
                    selected[phase * tile_bits + slot] = slot;
                }
            }
            for (int offset = 0; offset < counts[phase]; ++offset) {
                const int slot = first_slot + offset;
                selected[phase * tile_bits + slot] = target++;
                target_masks[phase] |= 1 << slot;
            }
        }
    } else if (layout == "low" || layout == "high" ||
               layout.rfind("range:", 0) == 0) {
        selected.assign(tile_bits, -1);
        target_masks.assign(1, 0);
        int first = layout == "low" ? 0 : qubits - tile_bits;
        if (layout.rfind("range:", 0) == 0) {
            try {
                first = std::stoi(layout.substr(6));
            } catch (...) {
                return 2;
            }
        }
        const int count = std::min(tile_bits, qubits - first);
        if (first < 0 || count <= 0) return 2;
        for (int slot = 0; slot < count; ++slot) {
            selected[slot] = first + slot;
            target_masks[0] |= 1 << slot;
        }
    } else if (fixed_layout || pair_layout) {
        selected.assign(tile_bits, -1);
        target_masks.assign(1, 0);
        for (int slot = 0; slot < kLaneBits; ++slot) selected[slot] = slot;
        const int first_warp_slot = kLaneBits + register_bits;
        if (pair_layout) {
            if (first_warp_slot >= tile_bits || qubits < kLaneBits + 1) return 2;
            selected[first_warp_slot] = kLaneBits;
        }
        const int high_count = tile_bits - kLaneBits - int(pair_layout);
        int first = qubits - high_count;
        const size_t prefix = pair_layout ? std::string("pairs:").size()
                                          : std::string("fixed:").size();
        if (layout.find(':') != std::string::npos) {
            try {
                first = std::stoi(layout.substr(prefix));
            } catch (...) {
                return 2;
            }
        }
        const int count = std::min(high_count, qubits - first);
        if (first < kLaneBits + int(pair_layout) || count <= 0) return 2;
        for (int offset = 0; offset < count; ++offset) {
            int slot = kLaneBits + offset;
            if (pair_layout && slot >= first_warp_slot) ++slot;
            selected[slot] = first + offset;
            target_masks[0] |= 1 << slot;
        }
    } else {
        return 2;
    }

    for (size_t phase = 0; phase < target_masks.size(); ++phase) {
        for (int slot = 0; slot < tile_bits; ++slot) {
            int& selected_qubit = selected[phase * tile_bits + slot];
            if (selected_qubit >= 0) continue;
            for (int qubit = 0; qubit < qubits; ++qubit) {
                bool used = false;
                for (int other = 0; other < tile_bits; ++other) {
                    used |= selected[phase * tile_bits + other] == qubit;
                }
                if (!used) {
                    selected_qubit = qubit;
                    break;
                }
            }
        }
    }

    if (target_spec != "all") {
        if (target_masks.size() != 1) return 2;
        int first_slot = 0;
        int count = 0;
        if (target_spec == "none") {
            count = 0;
        } else if (target_spec == "lane") {
            count = kLaneBits;
        } else if (target_spec == "register") {
            first_slot = kLaneBits;
            count = register_bits;
        } else if (target_spec == "warp") {
            first_slot = kLaneBits + register_bits;
            count = tile_bits - first_slot;
        } else if (target_spec.rfind("L", 0) == 0) {
            PhaseTargetClasses classes;
            char extra = 0;
            if (std::sscanf(target_spec.c_str(), "L%dR%dW%d%c",
                            &classes.lane,
                            &classes.reg,
                            &classes.warp,
                            &extra) != 3 ||
                classes.lane < 0 || classes.lane > kLaneBits ||
                classes.reg < 0 || classes.reg > register_bits ||
                classes.warp < 0 ||
                classes.warp > tile_bits - kLaneBits - register_bits ||
                ((fixed_layout || pair_layout) && classes.lane != 0) ||
                (pair_layout &&
                 classes.warp >
                     tile_bits - kLaneBits - register_bits - 1)) {
                return 2;
            }
            target_masks[0] = 0;
            for (int slot = 0; slot < classes.lane; ++slot) {
                target_masks[0] |= 1 << slot;
            }
            for (int offset = 0; offset < classes.reg; ++offset) {
                target_masks[0] |= 1 << (kLaneBits + offset);
            }
            for (int offset = 0; offset < classes.warp; ++offset) {
                target_masks[0] |=
                    1 << (kLaneBits + register_bits + offset);
            }
            count = -1;
        } else {
            try {
                count = std::stoi(target_spec);
            } catch (...) {
                return 2;
            }
            first_slot = fixed_layout || pair_layout ? kLaneBits : 0;
        }
        if (count < -1 || first_slot < 0 || first_slot + count > tile_bits) {
            return 2;
        }
        if (count >= 0) {
            target_masks[0] = 0;
            for (int slot = first_slot; slot < first_slot + count; ++slot) {
                target_masks[0] |= 1 << slot;
            }
        }
    }

    uint64_t warp_phase_mask = 0;
    for (size_t phase = 0; phase < target_masks.size(); ++phase) {
        if ((target_masks[phase] >> (kLaneBits + register_bits)) != 0) {
            warp_phase_mask |= 1ull << phase;
        }
    }

    const uint64_t state_size = 1ull << qubits;
    Complex<double>* phi = nullptr;
    Complex<double>* lambda = nullptr;
    RotationCoefficients<double>* coefficients = nullptr;
    double* gradients = nullptr;
    int* device_selected = nullptr;
    int* device_mask = nullptr;
    SAD_CUDA_CHECK(cudaMalloc(&phi, state_size * sizeof(*phi)));
    SAD_CUDA_CHECK(cudaMalloc(&lambda, state_size * sizeof(*lambda)));
    SAD_CUDA_CHECK(cudaMalloc(&coefficients, qubits * sizeof(*coefficients)));
    SAD_CUDA_CHECK(cudaMalloc(&gradients, qubits * sizeof(*gradients)));
    SAD_CUDA_CHECK(cudaMalloc(&device_selected, selected.size() * sizeof(int)));
    SAD_CUDA_CHECK(
        cudaMalloc(&device_mask, target_masks.size() * sizeof(int)));
    SAD_CUDA_CHECK(cudaMemset(phi, 0, state_size * sizeof(*phi)));
    SAD_CUDA_CHECK(cudaMemset(lambda, 0, state_size * sizeof(*lambda)));
    SAD_CUDA_CHECK(cudaMemset(gradients, 0, qubits * sizeof(*gradients)));
    std::vector<RotationCoefficients<double>> host_coefficients(
        qubits, {0.123, 0.99240677});
    SAD_CUDA_CHECK(cudaMemcpy(coefficients,
                              host_coefficients.data(),
                              qubits * sizeof(*coefficients),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(device_selected,
                              selected.data(),
                              selected.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(device_mask,
                              target_masks.data(),
                              target_masks.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    cudaDeviceProp properties{};
    SAD_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    cudaFuncAttributes attributes{};
    int active_blocks = 0;
    int no_mailbox_active_blocks = 0;
    size_t dynamic_shared_bytes = 0;
    size_t mailbox_bytes = 0;
    if (direction == "forward") {
        if (gate == "rx") {
            dynamic_shared_bytes =
                forward_rotation_mailbox_bytes<double, NonDiagonalGate::RX>();
            mailbox_bytes = dynamic_shared_bytes;
            if (dynamic_shared_bytes > 0) {
                SAD_CUDA_CHECK(cudaFuncSetAttribute(
                    non_diagonal_forward_kernel<double, NonDiagonalGate::RX>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(dynamic_shared_bytes)));
            }
            SAD_CUDA_CHECK(cudaFuncGetAttributes(
                &attributes,
                non_diagonal_forward_kernel<double, NonDiagonalGate::RX>));
            SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                non_diagonal_forward_kernel<double, NonDiagonalGate::RX>,
                kForwardBlockThreads,
                dynamic_shared_bytes));
            SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &no_mailbox_active_blocks,
                non_diagonal_forward_kernel<double, NonDiagonalGate::RX>,
                kForwardBlockThreads,
                0));
        } else {
            dynamic_shared_bytes =
                forward_rotation_mailbox_bytes<double, NonDiagonalGate::RY>();
            mailbox_bytes = dynamic_shared_bytes;
            if (dynamic_shared_bytes > 0) {
                SAD_CUDA_CHECK(cudaFuncSetAttribute(
                    non_diagonal_forward_kernel<double, NonDiagonalGate::RY>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(dynamic_shared_bytes)));
            }
            SAD_CUDA_CHECK(cudaFuncGetAttributes(
                &attributes,
                non_diagonal_forward_kernel<double, NonDiagonalGate::RY>));
            SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                non_diagonal_forward_kernel<double, NonDiagonalGate::RY>,
                kForwardBlockThreads,
                dynamic_shared_bytes));
            SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &no_mailbox_active_blocks,
                non_diagonal_forward_kernel<double, NonDiagonalGate::RY>,
                kForwardBlockThreads,
                0));
        }
    }
#ifndef SAD_MICRO_FORWARD_ONLY
    else if (gate == "rx") {
        mailbox_bytes =
            backward_rotation_mailbox_bytes<double, NonDiagonalGate::RX>();
        SAD_CUDA_CHECK(cudaFuncGetAttributes(
            &attributes,
            non_diagonal_backward_gradient_kernel<double,
                                                  NonDiagonalGate::RX>));
        SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            non_diagonal_backward_gradient_kernel<double,
                                                  NonDiagonalGate::RX>,
            kBlockThreads,
            0));
        SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &no_mailbox_active_blocks,
            non_diagonal_backward_gradient_kernel<
                double, NonDiagonalGate::RX, false, false>,
            kBlockThreads,
            0));
    } else {
        mailbox_bytes =
            backward_rotation_mailbox_bytes<double, NonDiagonalGate::RY>();
        SAD_CUDA_CHECK(cudaFuncGetAttributes(
            &attributes,
            non_diagonal_backward_gradient_kernel<double,
                                                  NonDiagonalGate::RY>));
        SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            non_diagonal_backward_gradient_kernel<double,
                                                  NonDiagonalGate::RY>,
            kBlockThreads,
            0));
        SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &no_mailbox_active_blocks,
            non_diagonal_backward_gradient_kernel<
                double, NonDiagonalGate::RY, false, false>,
            kBlockThreads,
            0));
    }
#endif
    auto launch = [&]() {
        if (direction == "forward") {
            if (gate == "rx") {
                launch_non_diagonal_forward<double, NonDiagonalGate::RX>(
                    phi,
                    coefficients,
                    qubits,
                    0,
                    device_selected,
                    device_mask,
                    static_cast<int>(target_masks.size()),
                    properties.multiProcessorCount,
                    false,
                    adaptive_mailbox ? warp_phase_mask : ~0ull);
            } else {
                launch_non_diagonal_forward<double, NonDiagonalGate::RY>(
                    phi,
                    coefficients,
                    qubits,
                    0,
                    device_selected,
                    device_mask,
                    static_cast<int>(target_masks.size()),
                    properties.multiProcessorCount,
                    false,
                    adaptive_mailbox ? warp_phase_mask : ~0ull);
            }
        }
#ifndef SAD_MICRO_FORWARD_ONLY
        else if (direction == "backward") {
            if (gate == "rx") {
                launch_non_diagonal_backward<double, NonDiagonalGate::RX>(
                    phi,
                    lambda,
                    coefficients,
                    gradients,
                    qubits,
                    0,
                    device_selected,
                    device_mask,
                    static_cast<int>(target_masks.size()),
                    properties.multiProcessorCount,
                    false,
                    adaptive_mailbox ? warp_phase_mask : ~0ull);
            } else {
                launch_non_diagonal_backward<double, NonDiagonalGate::RY>(
                    phi,
                    lambda,
                    coefficients,
                    gradients,
                    qubits,
                    0,
                    device_selected,
                    device_mask,
                    static_cast<int>(target_masks.size()),
                    properties.multiProcessorCount,
                    false,
                    adaptive_mailbox ? warp_phase_mask : ~0ull);
            }
        }
#endif
        else {
            throw std::runtime_error("direction must be forward or backward");
        }
    };
    for (int warmup = 0; warmup < 5; ++warmup) launch();
    SAD_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    SAD_CUDA_CHECK(cudaEventCreate(&start));
    SAD_CUDA_CHECK(cudaEventCreate(&stop));
    constexpr int iterations = SAD_MICRO_ITERATIONS;
    SAD_CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) launch();
    SAD_CUDA_CHECK(cudaEventRecord(stop));
    SAD_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    SAD_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    int gate_count = 0;
    std::vector<int> phase_targets;
    std::vector<int> phase_lane_targets;
    std::vector<int> phase_register_targets;
    std::vector<int> phase_warp_targets;
    for (const int mask : target_masks) {
        const int lane = __builtin_popcount(
            static_cast<unsigned>(mask & ((1 << kLaneBits) - 1)));
        const int reg = __builtin_popcount(static_cast<unsigned>(
            mask & (((1 << register_bits) - 1) << kLaneBits)));
        const int warp = __builtin_popcount(static_cast<unsigned>(
            mask >> (kLaneBits + register_bits)));
        phase_lane_targets.push_back(lane);
        phase_register_targets.push_back(reg);
        phase_warp_targets.push_back(warp);
        phase_targets.push_back(lane + reg + warp);
        gate_count += lane + reg + warp;
    }
    const double average_ms = elapsed_ms / iterations;
    const double per_gate_ms = gate_count == 0 ? 0.0 : average_ms / gate_count;
    const std::string measured_layout = csv_safe(
        target_spec == "all" ? layout : layout + ":" + target_spec);
    if (adaptive_mailbox) {
        std::printf(
            "adaptive_mailbox=1,no_mailbox_active_cta_per_sm=%d,"
            "warp_phase_mask=%llu\n",
            no_mailbox_active_blocks,
            static_cast<unsigned long long>(warp_phase_mask));
    }
    std::printf(
        "%s,%s,%s,%d,%d,%d,%d,%zu,%d,%d,%zu,%zu,%d,%.9f,%.9f,"
        "%zu,%d,%d,%d,%d,%zu,%d,%s,%s,%s,%s\n",
                gate.c_str(),
                measured_layout.c_str(),
                direction.c_str(),
                qubits,
                direction == "forward" ? kForwardBlockThreads
                                         : kBlockThreads,
                direction == "forward" ? kForwardRegisterAmplitudes
                                         : kRegisterAmplitudes,
                tile_bits,
                target_masks.size(),
                gate_count,
                attributes.numRegs,
                attributes.sharedSizeBytes,
                dynamic_shared_bytes,
                active_blocks,
                average_ms,
                per_gate_ms,
                mailbox_bytes,
                kMailboxChunks,
                kRyScalarMailbox ? 1 : 0,
                kRotationPersistent ? 1 : 0,
                kLegacyBlockReduction ? 1 : 0,
                attributes.localSizeBytes,
                properties.multiProcessorCount,
                join_ints(phase_targets, ':').c_str(),
                join_ints(phase_lane_targets, ':').c_str(),
                join_ints(phase_register_targets, ':').c_str(),
                join_ints(phase_warp_targets, ':').c_str());

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(phi);
    cudaFree(lambda);
    cudaFree(coefficients);
    cudaFree(gradients);
    cudaFree(device_selected);
    cudaFree(device_mask);
    return 0;
}
