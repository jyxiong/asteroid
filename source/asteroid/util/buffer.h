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

    template<typename T>
    class DeviceBuffer {
    public:

        explicit DeviceBuffer(size_t size = 0)
                : m_Size(size), m_Data(0) {
            alloc(size);
        }

        DeviceBuffer(const T *data, size_t size, std::vector<T>& hostData)
                : m_Size(size), m_Data(0) {
            alloc(size);
            cudaMemcpy(m_Data, data, sizeof(T) * m_Size, cudaMemcpyHostToDevice);

            CopyTo(hostData);
        }

        ~DeviceBuffer() {
            cudaFree(m_Data);

        }

        void CopyTo(std::vector<T>& hostData)
        {
            CUDA_CHECK(cudaMemcpy(hostData.data(), m_Data, sizeof(T) * m_Size, cudaMemcpyDeviceToHost);)
        }

    private:
        void alloc(size_t size)
        {
            if (m_Data)
            {
                free();
            }

            m_Size = size;
            if (size > 0)
            {
                CUDA_CHECK( cudaMalloc(&m_Data, sizeof(T) * m_Size) )
            }
        }


        void free()
        {
            if (m_Data)
            {
                CUDA_CHECK( cudaFree(m_Data) );
            }
            m_Data = 0;
            m_Size = 0;
        }


    private:
        T *m_Data;

        size_t m_Size;

        friend DeviceBufferView<T>;
    };

} // namespace name
