#pragma once

#include <string>

namespace Asteroid
{
// https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexStorage2D.xhtml
enum class InternalFormat
{
    None = 0,
    RGBA8,
    RGBA8UI,
};

// https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexSubImage2D.xhtml
enum class PixelFormat
{
    None = 0,
    RGBA,
    BGRA,
};

// https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexSubImage2D.xhtml
enum class PixelType
{
    None = 0,
    UNSIGNED_BYTE,
    UNSIGNED_INT,
};

struct TextureSpecification
{
    int Width = 1;
    int Height = 1;
    InternalFormat Format = InternalFormat::None;
    PixelFormat pixel_format = PixelFormat::None;
    bool GenerateMips = true;
};

class Texture2D
{
public:
    explicit Texture2D(const TextureSpecification &specification);

    Texture2D(const std::string &path);

    ~Texture2D();

    const TextureSpecification &GetSpecification() const { return m_Specification; }

    int GetWidth() const { return m_Width; }

    int GetHeight() const { return m_Height; }

    unsigned int GetRendererID() const { return m_RendererID; }

    const std::string &GetPath() const { return m_Path; }

    void SetData(void *data, unsigned int size) const;

    void Bind(unsigned int slot = 0) const;

    bool IsLoaded() const { return m_IsLoaded; }

private:
    TextureSpecification m_Specification;

    std::string m_Path;
    bool m_IsLoaded{false};
    int m_Width{}, m_Height{};
    unsigned int m_RendererID{};
    unsigned int m_InternalFormat{}, m_DataFormat{};
};

}