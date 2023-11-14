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

    explicit Image(const glm::ivec2& size);

    ~Image();

    void setData(const void* data);

    void onResize(const glm::ivec2& size);

    [[nodiscard]] int width() const { return m_size.x; }
    [[nodiscard]] int height() const { return m_size.y; }
    [[nodiscard]] unsigned int rendererID() const { return m_rendererID; }

private:
    void allocate();

    void release();

private:
    glm::ivec2 m_size{};

    unsigned int m_rendererID{};

    cudaGraphicsResource_t m_resource{};
};
}