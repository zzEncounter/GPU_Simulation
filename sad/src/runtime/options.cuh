#pragma once

#include "sad_api.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sad {

#ifndef SAD_REAL_AMPLITUDE
#define SAD_REAL_AMPLITUDE 1
#endif

enum class ExecutionMode : int {
    OPTIMIZED = 0,
    LEGACY = 1,
    ALL_FUSED = 2,
    INITIAL_ONLY = 3,
    FUSED_FORWARD = 4,
    PHASED_FORWARD = 5,
};

inline ExecutionMode read_execution_mode() {
    const char* value = std::getenv("SAD_EXECUTION_MODE");
    if (value == nullptr || *value == '\0' ||
        std::string(value) == "optimized") {
        return ExecutionMode::OPTIMIZED;
    }
    if (std::string(value) == "legacy") {
        return ExecutionMode::LEGACY;
    }
    if (std::string(value) == "all-fused") {
        return ExecutionMode::ALL_FUSED;
    }
    if (std::string(value) == "initial-only") {
        return ExecutionMode::INITIAL_ONLY;
    }
    if (std::string(value) == "fused-forward") {
        return ExecutionMode::FUSED_FORWARD;
    }
    if (std::string(value) == "phased-forward") {
        return ExecutionMode::PHASED_FORWARD;
    }
    throw std::invalid_argument(
        "SAD_EXECUTION_MODE must be optimized, legacy, initial-only, "
        "fused-forward, phased-forward, or all-fused");
}

inline bool use_real_amplitude_state(int circuit, ExecutionMode mode) {
    return SAD_REAL_AMPLITUDE != 0 && circuit == SAD_CIRCUIT_RA_HEA &&
           (mode == ExecutionMode::OPTIMIZED ||
            mode == ExecutionMode::ALL_FUSED);
}

}  // namespace sad
