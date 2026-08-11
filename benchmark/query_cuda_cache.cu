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
    std::printf("l2_bytes=%d\n", properties.l2CacheSize);
    std::printf("persisting_l2_max_bytes=%d\n",
                properties.persistingL2CacheMaxSize);
    std::printf("access_policy_window_max_bytes=%d\n",
                properties.accessPolicyMaxWindowSize);
    std::printf("configured_persisting_l2_bytes=%zu\n", configured_set_aside);
    return 0;
}
