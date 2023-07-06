#include "asteroid/opengl/texture2d.h"

#include "glad/gl.h"
#include "stb_image.h"
#include "asteroid/util/macro.h"

using namespace Asteroid;

static unsigned int GetGLPixelFormat(PixelFormat format)
{
    switch (format)
    {
        case PixelFormat::BGRA:
            return GL_BGRA;
        case PixelFormat::RGBA:
            return GL_RGBA;
        default: AST_CORE_ASSERT(false, "Unknown PixelFormat!")
            return 0;
    }
}

static unsigned int GetGLInternalFormat(InternalFormat format)
{
    switch (format)
    {
        case InternalFormat::RGBA8:
            return GL_RGBA8;
        case InternalFormat::RGBA8UI:
            return GL_RGBA8UI;
        default: AST_CORE_ASSERT(false, "Unknown InternalFormat!")
            return 0;
    }
}

static unsigned int GetGLPixelType(PixelType type)
{
    switch (type)
    {
    case PixelType::UNSIGNED_BYTE:
        return GL_UNSIGNED_BYTE;
    case PixelType::UNSIGNED_INT:
        return GL_UNSIGNED_INT;
    default: AST_CORE_ASSERT(false, "Unknown InternalFormat!")
        return 0;
    }
}

Texture2D::Texture2D(const TextureSpecification &specification)
    : m_Specification(specification), m_Width(m_Specification.Width), m_Height(m_Specification.Height)
{

    m_InternalFormat = GetGLInternalFormat(m_Specification.Format);
    m_DataFormat = GetGLPixelFormat(m_Specification.pixel_format);

    glCreateTextures(GL_TEXTURE_2D, 1, &m_RendererID);
    glTextureStorage2D(m_RendererID, 1, m_InternalFormat, m_Width, m_Height);

    glTextureParameteri(m_RendererID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_RendererID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTextureParameteri(m_RendererID, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(m_RendererID, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

Texture2D::Texture2D(const std::string &path)
    : m_Path(path)
{
    int width, height, channels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &channels, 0);

    if (data)
    {
        m_IsLoaded = true;

        m_Width = width;
        m_Height = height;

        GLenum internalFormat = 0, dataFormat = 0;
        if (channels == 4)
        {
            internalFormat = GL_RGBA8;
            dataFormat = GL_RGBA;
        } else if (channels == 3)
        {
            internalFormat = GL_RGB8;
            dataFormat = GL_RGB;
        }

        m_InternalFormat = internalFormat;
        m_DataFormat = dataFormat;

        AST_CORE_ASSERT(internalFormat & dataFormat, "Format not supported!")

        glCreateTextures(GL_TEXTURE_2D, 1, &m_RendererID);
        glTextureStorage2D(m_RendererID, 1, internalFormat, m_Width, m_Height);

        glTextureParameteri(m_RendererID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(m_RendererID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTextureParameteri(m_RendererID, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(m_RendererID, GL_TEXTURE_WRAP_T, GL_REPEAT);

        glTextureSubImage2D(m_RendererID, 0, 0, 0, m_Width, m_Height, dataFormat, GL_UNSIGNED_BYTE, data);

        stbi_image_free(data);
    }
}

Texture2D::~Texture2D()
{
    glDeleteTextures(1, &m_RendererID);
}

void Texture2D::SetData(void *data, unsigned int size) const
{/*
    unsigned int bpp = m_DataFormat == GL_RGBA ? 4 : 3;
    AST_CORE_ASSERT(size == m_Width * m_Height * bpp, "Data must be entire texture!")*/
    glTextureSubImage2D(m_RendererID, 0, 0, 0, m_Width, m_Height, m_DataFormat, GL_UNSIGNED_BYTE, data);
}

void Texture2D::Bind(unsigned int slot) const
{
    glBindTextureUnit(slot, m_RendererID);
}
