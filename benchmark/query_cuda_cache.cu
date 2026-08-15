#include <cuda_runtime.h>

#include <cstdio>

int main() {
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return 1;
    }
    std::size_t configured_set_aside = 0;
    if (cudaDeviceGetLimit(&configured_set_aside,
                           cudaLimitPersistingL2CacheSize) != cudaSuccess) {
        return 1;
    }
    std::printf("name=%s\n", properties.name);
    std::printf("compute_capability=%d.%d\n", properties.major, properties.minor);
    std::printf("multiprocessors=%d\n", properties.multiProcessorCount);
    std::printf("warp_size=%d\n", properties.warpSize);
    std::printf("max_threads_per_sm=%d\n",
                properties.maxThreadsPerMultiProcessor);
    std::printf("registers_per_sm=%d\n",
                properties.regsPerMultiprocessor);
    std::printf("shared_bytes_per_sm=%zu\n",
                properties.sharedMemPerMultiprocessor);
    std::printf("shared_bytes_per_block=%zu\n",
                properties.sharedMemPerBlockOptin);
    int memory_clock_khz = 0;
    int memory_bus_bits = 0;
    cudaDeviceGetAttribute(
        &memory_clock_khz, cudaDevAttrMemoryClockRate, device);
    cudaDeviceGetAttribute(
        &memory_bus_bits, cudaDevAttrGlobalMemoryBusWidth, device);
    std::printf("memory_clock_khz=%d\n", memory_clock_khz);
    std::printf("memory_bus_bits=%d\n", memory_bus_bits);
    std::printf("l2_bytes=%d\n", properties.l2CacheSize);
    std::printf("persisting_l2_max_bytes=%d\n",
                properties.persistingL2CacheMaxSize);
    std::printf("access_policy_window_max_bytes=%d\n",
                properties.accessPolicyMaxWindowSize);
    std::printf("configured_persisting_l2_bytes=%zu\n", configured_set_aside);
    return 0;
}
