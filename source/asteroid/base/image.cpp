#include "asteroid/base/image.h"

using namespace Asteroid;

Image::Image(const TextureSpecification& specification)
{
	m_Texture = std::make_shared<Texture2D>(specification);
	cudaGraphicsGLRegisterImage(&m_resource, m_Texture->GetRendererID(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

Image::~Image() = default;

void Image::SetData(const void* data, size_t size)
{
    cudaArray* texture_ptr;
    cudaGraphicsMapResources(1, &m_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_resource, 0, 0);
    cudaMemcpyToArray(texture_ptr, 0, 0, data, size, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &m_resource, 0);
}

void Image::Resize(unsigned int width, unsigned int height)
{
    if (m_Texture && m_Width == width && m_Height == height)
    {
		return;
    }

	// TODO: max size?
	m_Width = width;
	m_Height = height;
        
    auto spec = m_Texture->GetSpecification();
    spec.Width = width;
    spec.Height = height;
    m_Texture = std::make_shared<Texture2D>(spec);
    cudaGraphicsGLRegisterImage(&m_resource, m_Texture->GetRendererID(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

void Image::Regist()
{
   cudaGraphicsGLRegisterImage(&m_resource, m_Texture->GetRendererID(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

}

void Image::UnRegist()
{
    cudaGraphicsUnregisterResource(m_resource);
}
