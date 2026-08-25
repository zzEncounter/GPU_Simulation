#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using Wrapper = void (*)(int,
                         int,
                         void*,
                         void*,
                         const void*,
                         double*,
                         int,
                         const int*,
                         const int*,
                         int);

#define DECLARE_SHAPE(name)                                                   \
    extern "C" void sad_shape_##name##_launch(int,                           \
                                                int,                           \
                                                void*,                         \
                                                void*,                         \
                                                const void*,                   \
                                                double*,                       \
                                                int,                           \
                                                const int*,                    \
                                                const int*,                    \
                                                int)

DECLARE_SHAPE(t32r4m1);
DECLARE_SHAPE(t32r2m1);
DECLARE_SHAPE(t64r3m1);
DECLARE_SHAPE(t64r3m2);
DECLARE_SHAPE(t64r3m4);
DECLARE_SHAPE(t64r3m8);
DECLARE_SHAPE(t64r2m1);
DECLARE_SHAPE(t64r2m2);
DECLARE_SHAPE(t64r2m4);
DECLARE_SHAPE(t64r4m1);
DECLARE_SHAPE(t64r4m2);
DECLARE_SHAPE(t64r4m4);
DECLARE_SHAPE(t64r4m8);
DECLARE_SHAPE(t64r4m16);
DECLARE_SHAPE(t128r2m1);
DECLARE_SHAPE(t128r2m2);
DECLARE_SHAPE(t128r2m4);
DECLARE_SHAPE(t128r3m1);
DECLARE_SHAPE(t128r3m2);
DECLARE_SHAPE(t128r3m4);
DECLARE_SHAPE(t128r3m8);

struct Shape {
    int threads;
    int register_bits;
    int tile_bits;
    Wrapper launch;
};

const std::unordered_map<std::string, Shape> kShapes = {
    {"t32r2m1", {32, 2, 7, sad_shape_t32r2m1_launch}},
    {"t32r4m1", {32, 4, 9, sad_shape_t32r4m1_launch}},
    {"t64r3m1", {64, 3, 9, sad_shape_t64r3m1_launch}},
    {"t64r3m2", {64, 3, 9, sad_shape_t64r3m2_launch}},
    {"t64r3m4", {64, 3, 9, sad_shape_t64r3m4_launch}},
    {"t64r3m8", {64, 3, 9, sad_shape_t64r3m8_launch}},
    {"t64r2m1", {64, 2, 8, sad_shape_t64r2m1_launch}},
    {"t64r2m2", {64, 2, 8, sad_shape_t64r2m2_launch}},
    {"t64r2m4", {64, 2, 8, sad_shape_t64r2m4_launch}},
    {"t64r4m1", {64, 4, 10, sad_shape_t64r4m1_launch}},
    {"t64r4m2", {64, 4, 10, sad_shape_t64r4m2_launch}},
    {"t64r4m4", {64, 4, 10, sad_shape_t64r4m4_launch}},
    {"t64r4m8", {64, 4, 10, sad_shape_t64r4m8_launch}},
    {"t64r4m16", {64, 4, 10, sad_shape_t64r4m16_launch}},
    {"t128r2m1", {128, 2, 9, sad_shape_t128r2m1_launch}},
    {"t128r2m2", {128, 2, 9, sad_shape_t128r2m2_launch}},
    {"t128r2m4", {128, 2, 9, sad_shape_t128r2m4_launch}},
    {"t128r3m1", {128, 3, 10, sad_shape_t128r3m1_launch}},
    {"t128r3m2", {128, 3, 10, sad_shape_t128r3m2_launch}},
    {"t128r3m4", {128, 3, 10, sad_shape_t128r3m4_launch}},
    {"t128r3m8", {128, 3, 10, sad_shape_t128r3m8_launch}},
};

void check(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " +
                                 cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(expr) check((expr), #expr)

struct Coefficients {
    double sine;
    double cosine;
};

struct ComplexValue {
    double real;
    double imag;
};

struct Phase {
    std::string variant;
    std::string family;
    int first;
    int count;
    Shape shape;
    int* selected = nullptr;
    int* mask = nullptr;
};

std::vector<std::string> split(const std::string& value, char separator) {
    std::stringstream stream(value);
    std::string token;
    std::vector<std::string> result;
    while (std::getline(stream, token, separator)) result.push_back(token);
    return result;
}

std::vector<Phase> parse_schedule(const std::string& value, int qubits) {
    std::vector<Phase> result;
    int first = 0;
    for (const auto& token : split(value, ';')) {
        const auto fields = split(token, '/');
        if (fields.size() != 3 || !kShapes.count(fields[0])) {
            throw std::invalid_argument("invalid phase token " + token);
        }
        const int count = std::stoi(fields[2]);
        const Shape shape = kShapes.at(fields[0]);
        if (count < 1 || first + count > qubits) {
            throw std::invalid_argument("invalid phase target count");
        }
        const int capacity = fields[1] == "compact"
                                 ? shape.tile_bits
                             : fields[1] == "fixed"
                                 ? shape.tile_bits - 5
                             : fields[1] == "pairs"
                                 ? shape.tile_bits - 6
                                 : -1;
        const int reserved = fields[1] == "fixed" ? 5
                           : fields[1] == "pairs" ? 6
                                                   : 0;
        if (capacity < count || first < reserved) {
            throw std::invalid_argument("phase does not fit selected family");
        }
        result.push_back({fields[0], fields[1], first, count, shape});
        first += count;
    }
    if (first != qubits) throw std::invalid_argument("schedule must cover every qubit");
    return result;
}

void build_map(Phase* phase, int qubits) {
    const int tile_bits = phase->shape.tile_bits;
    const int register_bits = phase->shape.register_bits;
    std::vector<int> selected(tile_bits, -1);
    int mask = 0;
    int first_target_slot = 0;
    int reserved_warp_slot = -1;
    if (phase->family == "fixed" || phase->family == "pairs") {
        for (int bit = 0; bit < 5; ++bit) selected[bit] = bit;
        first_target_slot = 5;
    }
    if (phase->family == "pairs") {
        reserved_warp_slot = 5 + register_bits;
        selected[reserved_warp_slot] = 5;
    }
    int slot = first_target_slot;
    for (int offset = 0; offset < phase->count; ++offset) {
        while (slot == reserved_warp_slot) ++slot;
        if (slot >= tile_bits) throw std::invalid_argument("target mask overflow");
        selected[slot] = phase->first + offset;
        mask |= 1 << slot;
        ++slot;
    }
    for (int target_slot = 0; target_slot < tile_bits; ++target_slot) {
        if (selected[target_slot] >= 0) continue;
        for (int qubit = 0; qubit < qubits; ++qubit) {
            if (std::find(selected.begin(), selected.end(), qubit) == selected.end()) {
                selected[target_slot] = qubit;
                break;
            }
        }
    }
    CUDA_CHECK(cudaMalloc(&phase->selected, tile_bits * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&phase->mask, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(phase->selected,
                          selected.data(),
                          tile_bits * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        phase->mask, &mask, sizeof(int), cudaMemcpyHostToDevice));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::fprintf(stderr,
                     "usage: %s QUBITS rx|ry forward|backward ITERATIONS "
                     "variant/family/count;... [paired-schedule]\n",
                     argv[0]);
        return 2;
    }
    const int qubits = std::stoi(argv[1]);
    const int gate = std::string(argv[2]) == "rx" ? 0 : 1;
    const int direction = std::string(argv[3]) == "forward" ? 0 : 1;
    const int iterations = std::stoi(argv[4]);
    if ((gate != 0 && std::string(argv[2]) != "ry") ||
        (direction != 0 && std::string(argv[3]) != "backward") ||
        qubits < 4 || qubits > 30 || iterations < 1) {
        return 2;
    }
    auto phases = parse_schedule(argv[5], qubits);
    for (auto& phase : phases) build_map(&phase, qubits);
    std::vector<Phase> paired_phases;
    if (argc == 7) {
        paired_phases = parse_schedule(argv[6], qubits);
        for (auto& phase : paired_phases) build_map(&phase, qubits);
    }

    const uint64_t state_size = 1ull << qubits;
    void* phi = nullptr;
    void* lambda = nullptr;
    Coefficients* coefficients = nullptr;
    double* gradients = nullptr;
    CUDA_CHECK(cudaMalloc(&phi, state_size * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lambda, state_size * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&coefficients, qubits * sizeof(Coefficients)));
    CUDA_CHECK(cudaMalloc(&gradients, qubits * sizeof(double)));
    CUDA_CHECK(cudaMemset(phi, 0, state_size * 2 * sizeof(double)));
    CUDA_CHECK(cudaMemset(lambda, 0, state_size * 2 * sizeof(double)));
    CUDA_CHECK(cudaMemset(gradients, 0, qubits * sizeof(double)));
    const ComplexValue phi_seed{1.0, 0.0};
    const ComplexValue lambda_seed{0.5, -0.25};
    CUDA_CHECK(cudaMemcpy(phi, &phi_seed, sizeof(phi_seed), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(lambda, &lambda_seed, sizeof(lambda_seed), cudaMemcpyHostToDevice));
    std::vector<Coefficients> host_coefficients(qubits, {0.123, 0.99240677});
    CUDA_CHECK(cudaMemcpy(coefficients,
                          host_coefficients.data(),
                          qubits * sizeof(Coefficients),
                          cudaMemcpyHostToDevice));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    auto launch = [&](std::vector<Phase>& scheduled_phases) {
        if (direction == 0) {
            for (auto& phase : scheduled_phases) {
                phase.shape.launch(direction,
                                   gate,
                                   phi,
                                   lambda,
                                   coefficients,
                                   gradients,
                                   qubits,
                                   phase.selected,
                                   phase.mask,
                                   properties.multiProcessorCount);
            }
        } else {
            for (auto phase = scheduled_phases.rbegin();
                 phase != scheduled_phases.rend();
                 ++phase) {
                phase->shape.launch(direction,
                                    gate,
                                    phi,
                                    lambda,
                                    coefficients,
                                    gradients,
                                    qubits,
                                    phase->selected,
                                    phase->mask,
                                    properties.multiProcessorCount);
            }
        }
    };
    for (int warmup = 0; warmup < 3; ++warmup) {
        launch(phases);
        if (!paired_phases.empty()) launch(paired_phases);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    constexpr int kChecksumValues = 16;
    auto reset_state = [&]() {
        CUDA_CHECK(cudaMemset(phi, 0, state_size * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(lambda, 0, state_size * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(gradients, 0, qubits * sizeof(double)));
        CUDA_CHECK(
            cudaMemcpy(phi, &phi_seed, sizeof(phi_seed), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lambda,
                              &lambda_seed,
                              sizeof(lambda_seed),
                              cudaMemcpyHostToDevice));
    };
    auto checksums = [&]() {
        std::vector<ComplexValue> host_phi(kChecksumValues);
        std::vector<ComplexValue> host_lambda(kChecksumValues);
        std::vector<double> host_gradients(qubits);
        CUDA_CHECK(cudaMemcpy(host_phi.data(),
                              phi,
                              host_phi.size() * sizeof(ComplexValue),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_lambda.data(),
                              lambda,
                              host_lambda.size() * sizeof(ComplexValue),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_gradients.data(),
                              gradients,
                              host_gradients.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        std::array<double, 3> result{};
        for (int index = 0; index < kChecksumValues; ++index) {
            result[0] += (index + 1) *
                         (host_phi[index].real + 0.5 * host_phi[index].imag);
            result[1] += (index + 1) *
                         (host_lambda[index].real +
                          0.5 * host_lambda[index].imag);
        }
        for (int index = 0; index < qubits; ++index) {
            result[2] += (index + 1) * host_gradients[index];
        }
        return result;
    };
    auto print_result = [&](const std::vector<Phase>& scheduled_phases,
                            float average_ms,
                            const std::array<double, 3>& checksum,
                            const char* schedule) {
        std::printf("%d,%s,%s,%zu,%.9f,%.17g,%.17g,%.17g,%s\n",
                    qubits,
                    argv[2],
                    argv[3],
                    scheduled_phases.size(),
                    average_ms,
                    checksum[0],
                    checksum[1],
                    checksum[2],
                    schedule);
    };

    if (paired_phases.empty()) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int iteration = 0; iteration < iterations; ++iteration) launch(phases);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        print_result(phases, elapsed_ms / iterations, checksums(), argv[5]);
    } else {
        std::array<float, 2> elapsed_ms{};
        auto time_once = [&](std::vector<Phase>& scheduled_phases, int index) {
            CUDA_CHECK(cudaEventRecord(start));
            launch(scheduled_phases);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float value = 0;
            CUDA_CHECK(cudaEventElapsedTime(&value, start, stop));
            elapsed_ms[index] += value;
        };
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if ((iteration & 1) == 0) {
                time_once(phases, 0);
                time_once(paired_phases, 1);
            } else {
                time_once(paired_phases, 1);
                time_once(phases, 0);
            }
        }
        reset_state();
        launch(phases);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto first_checksum = checksums();
        reset_state();
        launch(paired_phases);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto second_checksum = checksums();
        print_result(
            phases, elapsed_ms[0] / iterations, first_checksum, argv[5]);
        print_result(paired_phases,
                     elapsed_ms[1] / iterations,
                     second_checksum,
                     argv[6]);
    }

    for (auto& phase : phases) {
        cudaFree(phase.selected);
        cudaFree(phase.mask);
    }
    for (auto& phase : paired_phases) {
        cudaFree(phase.selected);
        cudaFree(phase.mask);
    }
    cudaFree(phi);
    cudaFree(lambda);
    cudaFree(coefficients);
    cudaFree(gradients);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
