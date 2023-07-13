#include "glm/glm.hpp"
#include "asteroid/renderer/path_tracer.h"
#include "asteroid/renderer/path_tracer_kernel.h"

namespace Asteroid
{

__global__ void cudaProcess(const Camera& camera, glm::u8vec4 *g_odata, int width, int height)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    auto uv = glm::vec2(x, y) / glm::vec2(width, height) * 2.f - 1.f;

    Ray ray;
    GeneratePrimaryRay(camera, uv, ray);

    auto color = TraceRay(ray);
    g_odata[y * width + x] = ConvertToRGBA(color);
}

void launch_cudaProcess(const Camera& camera, glm::u8vec4 *g_odata, int width, int height)
{
    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    cudaProcess<<<grid, block>>>(camera, g_odata, width, height);
}

}
