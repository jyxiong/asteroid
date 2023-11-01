#pragma once

#include <memory>
#include <cuda_runtime.h>
#include "glad/gl.h"

namespace Asteroid
{
class Image
{
public:
    Image(int width, int height);

    ~Image();

    void SetData(const void* data);

    void Resize(int width, int height);

    int GetWidth() const { return m_Width; }

    int GetHeight() const { return m_Height; }

    unsigned int GetRendererID() const { return m_RendererID; }

private:
    void Allocate();

    void Release();

private:
    int m_Width;

    int m_Height;

    unsigned int m_RendererID{};

    cudaGraphicsResource_t m_resource;
};
}