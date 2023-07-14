#pragma once

#include "glm/glm.hpp"
#include "asteroid/renderer/camera.h"
#include "asteroid/renderer/path_tracer_kernel.h"

namespace Asteroid
{
    __global__ void GeneratePrimaryRay(const Camera camera, Ray* rays)
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= camera.m_ViewportWidth && y >= camera.m_ViewportHeight)
            return;

        auto uv = glm::vec2(x, y) / glm::vec2(camera.m_ViewportWidth, camera.m_ViewportHeight) * 2.f - 1.f;

        GeneratePrimaryRayKernel(camera, uv, rays[y * camera.m_ViewportWidth + x]);
    }

    __global__ void GetColor(const Ray* rays, glm::u8vec4 *g_odata, int width, int height)
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width && y >= height)
            return;

        auto color = TraceRay(rays[y * width + x]);
        g_odata[y * width + x] = glm::u8vec4(ConvertToRGBA(color));
    }
}