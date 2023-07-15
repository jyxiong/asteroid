#pragma once

#include <cuda_runtime.h>

namespace Asteroid
{
    
template<typename T> class Buffer;
template<typename T>
struct BufferView
{
    explicit BufferView(const Buffer<T>& buffer)
        : data(buffer.m_Data), size(buffer.m_Size) {}

    T* data;

    size_t size;
};

template<typename T>
class Buffer
{
public:

	explicit Buffer(size_t size = 0)
        : m_Size(size)
	{
		cudaMalloc(&m_Data, sizeof(T) * m_Size);
	}

    Buffer(const T* data, size_t size)
        : m_Size(size)
	{
		cudaMalloc(&m_Data, sizeof(T) * m_Size);
        cudaMemcpy(m_Data, data, sizeof(T) * size, cudaMemcpyHostToDevice);
	}

	~Buffer()
	{
		cudaFree(m_Data);
	}

    BufferView<T> View() const { return BufferView<T>(*this); }

private:

    T* m_Data;

    size_t m_Size;

    friend BufferView<T>;
};

} // namespace name
