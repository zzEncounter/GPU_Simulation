#pragma once

#include "ising_cuda_backend_internal.cuh"

#include <cstddef>

namespace standalone_backend {
namespace detail {

constexpr std::size_t DENSE_SCAN_MAX_DIM = 64;
constexpr std::size_t ROTATION_CHUNK_MAX_WIRES = 8;
constexpr std::size_t DENSE_ROTATION_CHUNK_MAX_WIRES = 3;
constexpr std::size_t ROTATION_CHUNK_REGISTER_TILE_START = 4;
constexpr std::size_t ROTATION_CHUNK_REGISTER_TILE_MIN_TILES =
    std::size_t{1} << 15;

struct RotationChunkCoeffs {
    double c[ROTATION_CHUNK_MAX_WIRES]{};
    double s[ROTATION_CHUNK_MAX_WIRES]{};
    double cos_half[ROTATION_CHUNK_MAX_WIRES]{};
    double sin_half[ROTATION_CHUNK_MAX_WIRES]{};
};

__host__ __device__ inline auto bit_is_set(std::size_t index,
                                           std::size_t wire) -> bool {
    return ((index >> wire) & 1U) != 0U;
}

} // namespace detail
} // namespace standalone_backend
