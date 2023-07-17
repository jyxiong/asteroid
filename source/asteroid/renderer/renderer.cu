#include "asteroid/renderer/renderer.h"
#include "asteroid/renderer/path_tracer.h"
#include "asteroid/util/macro.h"

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

    cudaFree(m_Rays);
    cudaMalloc((void**)&m_Rays, sizeof(Ray) * pixel_num);

    cudaFree(m_Intersections);
    cudaMalloc((void**)&m_Intersections, sizeof(Intersection) * pixel_num);
}

void Renderer::Render(const Scene& scene, const Camera& camera)
{
	auto width = m_FinalImage->GetWidth();
	auto height = m_FinalImage->GetHeight();

	auto sceneView = SceneView(scene);

	// Execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    GeneratePrimaryRay<<<grid, block>>>(camera, m_Rays);
    CUDA_SYNC_CHECK()

    ComputeIntersection<<<grid, block>>>(sceneView, m_Rays, width, height,m_Intersections);
    CUDA_SYNC_CHECK()

    PerPixel<<<grid, block>>>(sceneView, m_Rays, m_ImageData, width, height);
    CUDA_SYNC_CHECK()

	m_FinalImage->SetData(m_ImageData);
}
