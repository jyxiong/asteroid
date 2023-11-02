#pragma once

#include <memory>
#include <cuda_runtime.h>
#include "glad/gl.h"
#include "glm/glm.hpp"

namespace Asteroid
{
class Image
{
public:
    Image();

    Image(const glm::ivec2& size);

    ~Image();

    void setData(const void* data);

    void resize(const glm::ivec2& size);

    int width() const { return m_size.x; }
    int height() const { return m_size.y; }
    unsigned int rendererID() const { return m_rendererID; }

private:
    void allocate();

    void release();

private:
    glm::ivec2 m_size{};

    unsigned int m_rendererID{};

    cudaGraphicsResource_t m_resource;
};
}