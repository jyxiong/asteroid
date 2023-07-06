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

		void SetData(const void* data, size_t size);

	
		std::shared_ptr<Texture2D> m_Texture;
		cudaGraphicsResource_t m_resource;
	};
}