#pragma once

#include <memory>
#include "glad/gl.h"
#include "cuda_gl_interop.h"
#include "asteroid/opengl/texture2d.h"

namespace Asteroid
{
	class Image
	{
	public:
		explicit Image(const TextureSpecification& specification);

		~Image();

		void SetData(const void* data, size_t size);

		void Resize(unsigned int width, unsigned int);

		unsigned int GetRendererID() { return m_Texture->GetRendererID(); }

		void UnRegist();

		void Regist();

	private:
		unsigned int m_Width;

		unsigned int m_Height;
	
		std::shared_ptr<Texture2D> m_Texture;
		cudaGraphicsResource_t m_resource;
	};
}