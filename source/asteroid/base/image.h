#pragma once

#include <memory>
#include "glad/gl.h"
#include "cuda_gl_interop.h"

namespace Asteroid
{
	class Image
	{
	public:
		Image(unsigned int width, unsigned int height);

		~Image();

		void SetData(const void* data);

		void Resize(unsigned int width, unsigned int);

		int GetWidth() const { return m_Width; }

		int GetHeight() const { return m_Height; }

		unsigned int GetRendererID() const { return m_RendererID; }

	private:
		void Allocate();

		void Release();

	private:
		unsigned int m_Width;

		unsigned int m_Height;

		unsigned int m_RendererID{};

		cudaGraphicsResource_t m_resource;
	};
}