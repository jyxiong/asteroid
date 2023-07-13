#include "asteroid/renderer/renderer.h"
#include "asteroid/renderer/path_tracer.h"

using namespace Asteroid;

void Renderer::OnResize(unsigned int width, unsigned int height)
{
	if (m_FinalImage)
	{
		// No resize necessary
		if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == height)
			return;

		m_FinalImage->Resize(width, height);
	}
	else
	{
		m_FinalImage = std::make_shared<Image>(width, height);
	}

	auto pixel_num = width * height;

	cudaFree(m_ImageData);
    cudaMalloc((void**)&m_ImageData, sizeof(glm::u8vec4) * pixel_num);

    cudaFree(m_AccumulationData);
    cudaMalloc((void**)&m_AccumulationData, sizeof(glm::vec4) * pixel_num);

    cudaFree(m_Ray);
    cudaMalloc((void**)&m_Ray, sizeof(Ray) * pixel_num);
}

void Renderer::Render(const Camera& camera)
{
	auto width = m_FinalImage->GetWidth();
	auto height = m_FinalImage->GetHeight();

	// Execute the kernel

	launch_cudaProcess(camera, m_ImageData, width, height);

	m_FinalImage->SetData(m_ImageData);
}
