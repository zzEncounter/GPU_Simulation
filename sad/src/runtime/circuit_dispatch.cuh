#pragma once

#include "sad_api.h"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sad {

// Keep the runtime-to-template boundary in one place. Callers receive an
// integral_constant so all work inside a circuit remains compile-time
// specialised and layer loops contain no circuit branch.
template <typename Function>
decltype(auto) visit_circuit(int circuit, Function&& function) {
    switch (circuit) {
        case SAD_CIRCUIT_RA_HEA:
            return std::forward<Function>(function)(
                std::integral_constant<int, SAD_CIRCUIT_RA_HEA>{});
        case SAD_CIRCUIT_SU2_HEA:
            return std::forward<Function>(function)(
                std::integral_constant<int, SAD_CIRCUIT_SU2_HEA>{});
        case SAD_CIRCUIT_RZZ_HEA:
            return std::forward<Function>(function)(
                std::integral_constant<int, SAD_CIRCUIT_RZZ_HEA>{});
        case SAD_CIRCUIT_QAOA:
            return std::forward<Function>(function)(
                std::integral_constant<int, SAD_CIRCUIT_QAOA>{});
        case SAD_CIRCUIT_XXZ_HVA:
            return std::forward<Function>(function)(
                std::integral_constant<int, SAD_CIRCUIT_XXZ_HVA>{});
        case SAD_CIRCUIT_MERA:
            return std::forward<Function>(function)(
                std::integral_constant<int, SAD_CIRCUIT_MERA>{});
        case SAD_CIRCUIT_EQUIVARIANT_QNN:
            return std::forward<Function>(function)(
                std::integral_constant<int, SAD_CIRCUIT_EQUIVARIANT_QNN>{});
        case SAD_CIRCUIT_DATA_REUPLOADING:
            return std::forward<Function>(function)(
                std::integral_constant<int, SAD_CIRCUIT_DATA_REUPLOADING>{});
        case SAD_CIRCUIT_QAOA_NS:
            return std::forward<Function>(function)(
                std::integral_constant<int, SAD_CIRCUIT_QAOA_NS>{});
        default:
            throw std::invalid_argument("unknown circuit id");
    }
}

}  // namespace sad
