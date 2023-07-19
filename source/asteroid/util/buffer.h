#pragma once

#include <cuda_runtime.h>
#include <asteroid/util/macro.h>

namespace Asteroid {

    template<typename T>
    class DeviceBuffer;

    template<typename T>
    class DeviceBufferView {
    public:
        explicit DeviceBufferView(const DeviceBuffer<T> &buffer)
                : m_data(buffer.m_Data), m_size(buffer.m_Size) {}

        __device__ size_t size() const { return m_size; }

        __host__ __device__ T *data() const { return m_data; }

        __device__ const T &operator[](size_t i) const {
            return m_data[i];
        }

        __device__ T &operator[](size_t i) {
            return m_data[i];
        }

    private:

        T *m_data;

        size_t m_size;
    };

    // https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Utils/CudaUtils.cpp#L72
    template<typename T>
    class DeviceBuffer {
    public:

        explicit DeviceBuffer(size_t size)
                : m_Data(nullptr), m_Size(size) {
            cudaMalloc(&m_Data, sizeof(T) * m_Size);
        }

        explicit DeviceBuffer(const std::vector<T>& buffer)
                : m_Data(nullptr), m_Size(buffer.size()) {
            cudaMalloc(&m_Data, sizeof(T) * m_Size);
            cudaMemcpy(m_Data, buffer.data(), sizeof(T) * buffer.size(), cudaMemcpyHostToDevice);
        }

        DeviceBuffer<T> &operator=(const DeviceBuffer<T> &buffer) = delete;

        ~DeviceBuffer() {
            cudaFree(m_Data);
        }

        size_t size() const { return m_Size; }

        T *data() const { return m_Data; }

    protected:
        T *m_Data;
        size_t m_Size;

        friend DeviceBufferView<T>;
    };

} // namespace name
