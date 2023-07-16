#pragma once

#include <cuda_runtime.h>
#include <asteroid/util/macro.h>

namespace Asteroid {

    template<typename T> class DeviceBuffer;

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
        DeviceBuffer(size_t count = 0) :
                m_Data(0), m_Size(count) {
            alloc(count);
        }

        DeviceBuffer(const std::vector<T> buffer)
                : m_Data(0), m_Size(buffer.size()) {
            alloc(m_Size);
            copyFrom(buffer.size(), buffer.data());
        }

        DeviceBuffer<T> &operator=(const DeviceBuffer<T> &buffer) = delete;

        ~DeviceBuffer() {
            free();
        }

        void alloc(size_t count) {
            if (m_Data) {
                free();
            }

            m_Size = count;


            if (count > 0) {

                CUDA_CHECK(cudaMalloc(&m_Data, sizeInByte()));

            }

            clear();
        }

        void free() {
            if (m_Data) {

                CUDA_CHECK(cudaFree(m_Data));
            }
            m_Data = 0;
            m_Size = 0;
        }

        void resize(const size_t count) {
            DeviceBuffer buffer(count);
            buffer.copyFrom(std::min(count, m_Size), m_Data);

            std::swap(m_Data, buffer.m_Data);
            std::swap(m_Size, buffer.m_Size);
        }

        void copyFrom(const size_t count, const T *src_ptr) {
            if (count == 0) {
                return;
            }

            if (count > m_Size) {
                alloc(count);
            }

            CUDA_CHECK(cudaMemcpy(m_Data, src_ptr, sizeof(T) * count, cudaMemcpyHostToDevice));

        }

        void clear() {
            if (m_Data) {

                CUDA_CHECK(cudaMemset(m_Data, 0, sizeInByte()));

            }
        }

        T operator[](const size_t i) const {

            T t;
            CUDA_CHECK(cudaMemcpy(&t, m_Data + i, sizeof(T), cudaMemcpyDeviceToHost));
            return t;

        }

        T &operator[](const size_t i) {

            T t;
            CUDA_CHECK(cudaMemcpy(&t, m_Data + i, sizeof(T), cudaMemcpyDeviceToHost));
            return t;

        }

        size_t size() const { return m_Size; }

        T *data() const { return m_Data; }

    protected:
        size_t sizeInByte() const { return m_Size * sizeof(T); }

    protected:
        T *m_Data;
        size_t m_Size;

        friend DeviceBufferView<T>;
    };

} // namespace name
