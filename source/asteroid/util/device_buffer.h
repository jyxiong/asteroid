#pragma once

#include <cuda_runtime.h>
#include "asteroid/util/macro.h"

namespace Asteroid
{

struct DeviceBuffer
{
    inline void *devicePtr() const
    {
        return m_devicePtr;
    }

    void resize(size_t size)
    {
        if (m_devicePtr) free();
        alloc(size);
    }

    void alloc(size_t size)
    {
//        AST_CORE_ASSERT(m_devicePtr == nullptr)
        m_sizeInBytes = size;
        AST_CUDA_CHECK(cudaMalloc((void **) &m_devicePtr, m_sizeInBytes))
    }

    void free()
    {
        AST_CUDA_CHECK(cudaFree(m_devicePtr))
        m_devicePtr = nullptr;
        m_sizeInBytes = 0;
    }

    template<typename T>
    void allocAndUpload(const std::vector<T> &vt)
    {
        alloc(vt.size() * sizeof(T));
        upload((const T *) vt.data(), vt.size());
    }

    template<typename T>
    void upload(const T *t, size_t count)
    {
//        AST_ASSERT(m_devicePtr != nullptr)
//        AST_ASSERT(m_sizeInBytes == count * sizeof(T))
        AST_CUDA_CHECK(cudaMemcpy(m_devicePtr, (void *) t,
                                  count * sizeof(T), cudaMemcpyHostToDevice))
    }

    template<typename T>
    void download(T *t, size_t count)
    {
//        AST_ASSERT(m_devicePtr != nullptr)
//        AST_ASSERT(m_sizeInBytes == count * sizeof(T))
        AST_CUDA_CHECK(cudaMemcpy((void *) t, m_devicePtr,
                                  count * sizeof(T), cudaMemcpyDeviceToHost))
    }

    size_t m_sizeInBytes{ 0 };
    void *m_devicePtr{ nullptr };
};
} // namespace name
