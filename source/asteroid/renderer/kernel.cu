#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "glm/glm.hpp"
#include "lib.h"

__global__ void cudaProcess(glm::u8vec4* g_odata, int width, int height) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	auto uv = glm::vec2(x, y) / glm::vec2(width, height) * 2.f - 1.f;

    g_odata[y * width + x] = ConvertToRGBA(PerPixel(uv));
}

void launch_cudaProcess(dim3 grid, dim3 block, glm::u8vec4* g_odata, int width, int height) {
    cudaProcess<<<grid, block>>>(g_odata, width, height);
}