#include "asteroid/opengl/vertex_array.h"

#include "glad/gl.h"
#include "asteroid/core/core.h"

using namespace Asteroid;

static GLenum ShaderDataTypeToOpenGLBaseType(ShaderDataType type)
{
    switch (type)
    {
        case Asteroid::ShaderDataType::Float:
            return GL_FLOAT;
        case Asteroid::ShaderDataType::Float2:
            return GL_FLOAT;
        case Asteroid::ShaderDataType::Float3:
            return GL_FLOAT;
        case Asteroid::ShaderDataType::Float4:
            return GL_FLOAT;
        case Asteroid::ShaderDataType::Mat3:
            return GL_FLOAT;
        case Asteroid::ShaderDataType::Mat4:
            return GL_FLOAT;
        case Asteroid::ShaderDataType::Int:
            return GL_INT;
        case Asteroid::ShaderDataType::Int2:
            return GL_INT;
        case Asteroid::ShaderDataType::Int3:
            return GL_INT;
        case Asteroid::ShaderDataType::Int4:
            return GL_INT;
        case Asteroid::ShaderDataType::Bool:
            return GL_BOOL;
        default: AST_CORE_ASSERT(false, "Unknown ShaderDataType!");
            return 0;
    }
}

VertexArray::VertexArray()
{
    glCreateVertexArrays(1, &m_RendererID);
}

VertexArray::~VertexArray()
{
    glDeleteVertexArrays(1, &m_RendererID);
}

void VertexArray::Bind() const
{
    glBindVertexArray(m_RendererID);
}

void VertexArray::Unbind() const
{
    glBindVertexArray(0);
}

void VertexArray::AddVertexBuffer(const std::shared_ptr<VertexBuffer> &vertexBuffer)
{
    AST_CORE_ASSERT(vertexBuffer->GetLayout().GetElements().size(), "Vertex Buffer has no layout!")

    glBindVertexArray(m_RendererID);
    vertexBuffer->Bind();

    unsigned int index = 0;
    const auto &layout = vertexBuffer->GetLayout();
    for (const auto &element: layout)
    {
        glEnableVertexAttribArray(index);
        glVertexAttribPointer(index,
                              element.GetComponentCount(),
                              ShaderDataTypeToOpenGLBaseType(element.Type),
                              element.Normalized ? GL_TRUE : GL_FALSE,
                              layout.GetStride(),
                              (const void *) element.Offset);
        index++;
    }
    m_VertexBuffers.push_back(vertexBuffer);
}

void VertexArray::SetIndexBuffer(const std::shared_ptr<IndexBuffer> &indexBuffer)
{
    glBindVertexArray(m_RendererID);
    indexBuffer->Bind();

    m_IndexBuffer = indexBuffer;
}
    