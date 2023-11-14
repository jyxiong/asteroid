#include "asteroid/app/image.h"
#include <cuda_gl_interop.h>

using namespace Asteroid;

Image::Image()
    : m_size(0)
{
    allocate();
}

Image::Image(const glm::ivec2& size)
    : m_size(size)
{
    allocate();
}

Image::~Image()
{
    release();
}

void Image::setData(const void* data)
{
    cudaGraphicsMapResources(1, &m_resource);

    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, m_resource, 0, 0);

    cudaMemcpy2DToArray(array,
                        0,
                        0,
                        data,
                        m_size.x * sizeof(glm::vec4),
                        m_size.x * sizeof(glm::vec4),
                        m_size.y,
                        cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &m_resource);
}

void Image::onResize(const glm::ivec2& size)
{
    if (m_size == size)
    {
        return;
    }

    release();

    m_size = size;

    allocate();
}

void Image::allocate()
{
    glCreateTextures(GL_TEXTURE_2D, 1, &m_rendererID);
    glTextureStorage2D(m_rendererID, 1, GL_RGBA32F, m_size.x, m_size.y);

    glTextureParameteri(m_rendererID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_rendererID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTextureParameteri(m_rendererID, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(m_rendererID, GL_TEXTURE_WRAP_T, GL_REPEAT);

    cudaGraphicsGLRegisterImage(&m_resource, m_rendererID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

void Image::release()
{
    if (!m_resource)
        cudaGraphicsUnregisterResource(m_resource);

    if (m_rendererID != 0)
        glDeleteTextures(1, &m_rendererID);
}
