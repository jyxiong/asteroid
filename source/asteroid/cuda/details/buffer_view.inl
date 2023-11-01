#pragma once

#include <cuda_runtime.h>

namespace Asteroid
{
template<typename T>
BufferView<T>::BufferView(T* data, size_t size)
    : m_data(data), m_size(size) {}

template<typename T>
__device__ const T& BufferView<T>::operator[](size_t i) const
{
    return m_data[i];
}

template<typename T>
__device__ T& BufferView<T>::operator[](size_t i)
{
    return m_data[i];
}

} // namespace name
