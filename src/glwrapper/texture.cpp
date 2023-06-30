#include "glwrapper/texture.h"
#include "glad/gl.h"

Texture2D::Texture2D()
{
    // 创建纹理
    glCreateTextures(GL_TEXTURE_2D, 1, &m_id);
    // 绑定纹理
    glBindTexture(GL_TEXTURE_2D, m_id);
    // 分配内存
    glTextureStorage2D(m_id, 1, GL_RGBA8, m_resolution.x, m_resolution.y);
    // 采样方式
    glTextureParameteri(m_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // 溢出处理方式
    glTextureParameteri(m_id, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(m_id, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

Texture2D::~Texture2D()
{
    glDeleteTextures(1, &m_id);
}

void Texture2D::bind(unsigned int unit) const
{
    glBindTextureUnit(unit, m_id);
}
