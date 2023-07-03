#pragma once

#include <utility>
#include "asteroid/util/macro.h"

namespace Asteroid
{

class IndexBuffer
{
public:
    IndexBuffer(unsigned int *indices, size_t count);

    ~IndexBuffer();

    void Bind() const;

    void Unbind() const;

    size_t GetCount() const { return m_Count; }

private:
    unsigned int m_RendererID{};
    unsigned int m_Count;
};

}