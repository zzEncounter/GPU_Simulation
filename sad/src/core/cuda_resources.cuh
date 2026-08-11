#pragma once

#include "cuda_common.cuh"

#include <chrono>
#include <cstddef>

namespace sad {

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : pointer_(other.pointer_), count_(other.count_) {
        other.pointer_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            pointer_ = other.pointer_;
            count_ = other.count_;
            other.pointer_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    ~DeviceBuffer() { release(); }

    void allocate(size_t count) {
        release();
        count_ = count;
        SAD_CUDA_CHECK(cudaMalloc(&pointer_, count * sizeof(T)));
    }
    T* get() const { return pointer_; }
    size_t bytes() const { return count_ * sizeof(T); }

  private:
    void release() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
            pointer_ = nullptr;
        }
    }
    T* pointer_ = nullptr;
    size_t count_ = 0;
};

template <typename T>
class PinnedBuffer {
  public:
    explicit PinnedBuffer(size_t count) {
        SAD_CUDA_CHECK(cudaMallocHost(&pointer_, count * sizeof(T)));
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() {
        if (pointer_ != nullptr) {
            cudaFreeHost(pointer_);
        }
    }
    T* get() const { return pointer_; }

  private:
    T* pointer_ = nullptr;
};

struct EventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    EventPair() {
        SAD_CUDA_CHECK(cudaEventCreate(&start));
        SAD_CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~EventPair() {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
    }
    template <typename Function>
    double measure(Function&& function) {
        const auto wall_start = std::chrono::steady_clock::now();
        SAD_CUDA_CHECK(cudaEventRecord(start));
        function();
        SAD_CUDA_CHECK(cudaEventRecord(stop));
        SAD_CUDA_CHECK(cudaEventSynchronize(stop));
        const auto wall_stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(wall_stop - wall_start).count();
    }
};


}  // namespace sad
