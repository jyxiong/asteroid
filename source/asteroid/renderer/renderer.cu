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
    CUDA_CHECK(cudaMalloc((void**)&m_Rays, sizeof(Ray) * pixel_num))
}

__global__ void test(glm::u8vec4 *g_odata, int width, int height)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width && y >= height)
        return;
    g_odata[y * width + x] = glm::u8vec4(255);
}

void Renderer::Render(const Camera& camera)
{
	auto width = m_FinalImage->GetWidth();
	auto height = m_FinalImage->GetHeight();

	// Execute the kernel

    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    CUDA_SYNC_CHECK()

    GeneratePrimaryRay<<<grid, block>>>(camera, m_Rays);
    CUDA_SYNC_CHECK()

    GetColor<<<grid, block>>>(m_Rays, m_ImageData, width, height);
    CUDA_SYNC_CHECK()

//    test<<<grid, block>>>(m_ImageData, width, height);
//    CUDA_SYNC_CHECK()

	m_FinalImage->SetData(m_ImageData);
}
