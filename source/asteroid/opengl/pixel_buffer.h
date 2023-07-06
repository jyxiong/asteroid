#pragma once

#include <utility>
#include "asteroid/util/macro.h"
namespace Asteroid
{
    class PixelBuffer
    {
    public:
        PixelBuffer(size_t size);

        ~PixelBuffer();

        unsigned int GetRendererID() const { return m_RendererID; }

        void BindPack() const;

        void UnbindPack() const;

        void BindUnpack() const;

        void UnbindUnpack() const;

    private:
        uint32_t m_RendererID;
    };
}
