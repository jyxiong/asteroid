#include "asteroid/renderer/renderer.h"
#include "asteroid/util/helper_cuda.h"

using namespace Asteroid;

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
    glm::u8vec4* g_odata, int imgw);

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

	int num_texels = width * height;


	checkCudaErrors(cudaFree(m_ImageData));
	checkCudaErrors(cudaMalloc((void**)&m_ImageData, sizeof(glm::u8vec4) * num_texels));

	checkCudaErrors(cudaFree(m_AccumulationData));
	checkCudaErrors(cudaMalloc((void**)&m_AccumulationData, sizeof(glm::vec4) * num_texels));
}

void Renderer::Render()
{
    auto width = m_FinalImage->GetWidth();
    auto height = m_FinalImage->GetHeight();

    // Execute the kernel
    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    launch_cudaProcess(grid, block, 0, m_ImageData, width);

	m_FinalImage->SetData(m_ImageData);
}
