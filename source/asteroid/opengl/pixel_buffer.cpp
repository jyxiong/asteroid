#include "asteroid/opengl/pixel_buffer.h"

#include "glad/gl.h"

using namespace Asteroid;

PixelBuffer::PixelBuffer(size_t size)
{
    glCreateBuffers(1, &m_RendererID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_RendererID);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_DYNAMIC_COPY);
}

PixelBuffer::~PixelBuffer()
{
    glDeleteBuffers(1, &m_RendererID);
}

void PixelBuffer::BindPack() const
{
    glBindBuffer(GL_PIXEL_PACK_BUFFER, m_RendererID);
}

void PixelBuffer::UnbindPack() const
{
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void PixelBuffer::BindUnpack() const
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_RendererID);
}

void PixelBuffer::UnbindUnpack() const
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
