#pragma once

#include <memory>
#include <cuda_runtime.h>

namespace Asteroid {

template<typename T>
struct CudaDeleter {
    constexpr CudaDeleter() noexcept = default;

    void operator()(T *ptr) const {
        cudaFree(ptr);
    }
};

template<typename T>
using unique_ptr = std::unique_ptr<T[], CudaDeleter<T> >;

template<typename T>
unique_ptr<T>
make_unique(size_t size) {
    T *ptr;
    cudaMalloc((void **) &ptr, size * sizeof(T));
    return unique_ptr<T>(ptr);
}

}
