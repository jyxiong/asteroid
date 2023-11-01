#pragma once

#include <cuda_runtime.h>
#include "asteroid/cuda/buffer_view.h"

namespace Asteroid
{
template<typename T>
class DeviceBuffer
{
protected:
    T* m_data;
    size_t m_size;
    size_t m_capacity;

public:
    DeviceBuffer();
    explicit DeviceBuffer(size_t size);
    explicit DeviceBuffer(const std::vector<T>& buffer);
    ~DeviceBuffer();

    DeviceBuffer<T>& operator=(const DeviceBuffer<T>& buffer) = delete;

    size_t size() const { return m_size; }
    T* data() const { return m_data; }

    void resize(size_t size);
    void clear();

    void upload(const T* buffer, size_t size);
    void upload(const std::vector<T>& buffer);
    void download(std::vector<T>& buffer);

    BufferView<T> view();

};
} // namespace name

#include "details/device_buffer.inl"
