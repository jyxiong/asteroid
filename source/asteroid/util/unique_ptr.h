#pragma once

#include <memory>
#include <cuda_runtime.h>

// https://onihusube.hatenablog.com/entry/2018/06/21/021550
// https://github.com/NVIDIA/TensorRT/blob/main/plugin/common/bertCommon.h#L293
// https://github.com/Kitware/kwiver/blob/master/arrows/cuda/cuda_memory.h#L20

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
