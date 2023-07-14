#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "glm/glm.hpp"
#include "asteroid/renderer/camera.h"
#include "asteroid/renderer/path_tracer_kernel.h"

namespace Asteroid
{
    __global__ void GeneratePrimaryRay(const Camera camera, Ray* rays)
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        auto viewport = camera.GetViewport();

        if (x >= viewport.x && y >= viewport.y)
            return;

        auto uv = glm::vec2(x, y) / glm::vec2(viewport) * 2.f - 1.f;

        camera.GeneratePrimaryRay(uv, rays[y * viewport.x + x]);
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