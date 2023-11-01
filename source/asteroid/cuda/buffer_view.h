#pragma once

#include <cuda_runtime.h>

namespace Asteroid
{
    template <typename T>
    class BufferView
    {
    private:
        T *m_data;

        size_t m_size;
    public:
        BufferView(T *data, size_t size);

        __device__ size_t size() const { return m_size; }
        __device__ T *data() const { return m_data; }

        __device__ const T &operator[](size_t i) const;
        __device__ T &operator[](size_t i);
    };

} // namespace name

#include "details/buffer_view.inl"
