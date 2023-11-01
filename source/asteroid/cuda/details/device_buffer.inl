#pragma once

#include <cuda_runtime.h>

namespace Asteroid
{
template<typename T>
DeviceBuffer<T>::DeviceBuffer()
    : m_data(nullptr), m_size(0), m_capacity(0)
{
}

template<typename T>
DeviceBuffer<T>::DeviceBuffer(size_t size)
    : m_data(nullptr), m_size(size), m_capacity(size)
{
    cudaMalloc(&m_data, sizeof(T) * m_size);
}

template<typename T>
DeviceBuffer<T>::DeviceBuffer(const std::vector<T>& buffer)
    : m_data(nullptr), m_size(buffer.size()), m_capacity(buffer.size())
{
    cudaMalloc(&m_data, sizeof(T) * m_size);
    cudaMemcpy(m_data, buffer.data(), sizeof(T) * buffer.size(), cudaMemcpyHostToDevice);
}

template<typename T>
DeviceBuffer<T>::~DeviceBuffer()
{
    cudaFree(m_data);
}

template<typename T>
void DeviceBuffer<T>::resize(size_t size)
{
    if (size > m_capacity)
    {
        cudaFree(m_data);
        cudaMalloc(&m_data, sizeof(T) * size);
        m_capacity = size;
    }
    m_size = size;
}

template<typename T>
void DeviceBuffer<T>::clear()
{
    cudaMemset(m_data, 0, sizeof(T) * m_size);
}

template<typename T>
void DeviceBuffer<T>::upload(const T* buffer, size_t size)
{
    resize(size);
    cudaMemcpy(m_data, buffer, sizeof(T) * size, cudaMemcpyHostToDevice);
}

template<typename T>
void DeviceBuffer<T>::upload(const std::vector<T>& buffer)
{
    upload(buffer.data(), buffer.size());
}

template<typename T>
void DeviceBuffer<T>::download(std::vector<T>& buffer)
{
    buffer.resize(m_size);
    cudaMemcpy(buffer.data(), m_data, sizeof(T) * m_size, cudaMemcpyDeviceToHost);
}

template<typename T>
BufferView<T> DeviceBuffer<T>::view()
{
    return { m_data, m_size };
}

} // namespace Asteroid
