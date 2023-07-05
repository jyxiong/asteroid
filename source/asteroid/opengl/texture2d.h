#pragma once

#include <string>

namespace Asteroid
{

enum class ImageFormat
{
    None = 0,
    R8,
    RGB8,
    RGBA8,
    RGBA8UI,
    RGBA32F
};

struct TextureSpecification
{
    int Width = 1;
    int Height = 1;
    ImageFormat Format = ImageFormat::RGBA8;
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