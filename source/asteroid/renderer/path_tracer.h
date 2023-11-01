#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "glm/glm.hpp"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/kernel/intersection.h"
#include "path_tracer_kernel.h"

namespace Asteroid
{
__global__ void
GeneratePathSegment(const Camera camera, unsigned int traceDepth, BufferView<PathSegment> pathSegments)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    auto viewport = camera.viewport;

    if (x >= viewport.x && y >= viewport.y)
        return;

    auto& path = pathSegments[y * viewport.x + x];
    path.color = glm::vec3(0);
    path.throughput = glm::vec3(1);
    path.remainingBounces = traceDepth;
    path.pixelIndex = y * viewport.x + x;

    auto uv = glm::vec2(x, y) / glm::vec2(viewport) * 2.f - 1.f;
    auto offsetX = float(uv.x) * camera.tanHalfFov * camera.aspectRatio * camera.right;
    auto offsetY = float(uv.y) * camera.tanHalfFov * camera.up;

    path.ray.direction = glm::normalize(camera.direction + offsetX + offsetY);
    path.ray.origin = camera.position;
}

__global__ void
ComputeIntersection(const SceneView scene, BufferView<PathSegment> pathSegments, int width, int height,
                    BufferView<Intersection> intersections)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width && y >= height)
        return;

    if (pathSegments[y * width + x].remainingBounces == 0)
    {
        return;
    }

    auto& path = pathSegments[y * width + x];

    int closestSphere = -1;
    Intersection its;
    float hitDistance = std::numeric_limits<float>::max();
    for (size_t i = 0; i < scene.deviceGeometries.size(); ++i)
    {
        const auto& geometry = scene.deviceGeometries[i];

        if (geometry.type == GeometryType::Sphere)
        {
            if (!intersect_sphere(geometry, path.ray, its))
            {
                continue;
            }
        }

        if (its.t < hitDistance && its.t > 0)
        {
            hitDistance = its.t;
            closestSphere = i;
        }
    }

    if (closestSphere < 0)
    {
        path.remainingBounces = 0;
        intersections[y * width + x].t = -1;
    } else
    {
        intersections[y * width + x] = its;
    }
}

__global__ void
Shading(const SceneView scene, BufferView<PathSegment> pathSegments, const BufferView<Intersection> its,
        int width, int height)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width && y >= height)
        return;

    if (pathSegments[y * width + x].remainingBounces == 0)
    {
        return;
    }

    auto& it = its[y * width + x];
    auto& material = scene.deviceMaterials[it.materialIndex];
    auto& path = pathSegments[y * width + x];

    if (material.emittance > 0.0f)
    {
        path.color += (material.albedo * material.emittance) * path.throughput;
        path.remainingBounces = 0;
    } else
    {
        scatterRay(path, it, material);
    }

//    if (path.pixelIndex == 142821)
//        printf("pixel index = %d, color = (%f, %f, %f); \n", path.pixelIndex, path.color.x, path.color.y, path.color.z);
}

__global__ void finalGather(BufferView<glm::vec3> image,
                            const BufferView<PathSegment> pathSegments,
                            int width,
                            int height)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width && y >= height)
        return;

    auto& path = pathSegments[y * width + x];
    image[y * width + x] += path.color;
}

__global__ void
ConvertToRGBA(const BufferView<glm::vec3> accumulations,
              unsigned int iter,
              int width,
              int height,
              BufferView<glm::u8vec4> image)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width && y >= height)
        return;

    auto color = glm::clamp(accumulations[y * width + x] / float(iter), 0.f, 1.f);
    image[y * width + x] = glm::u8vec4(color * 255.f, 255);
}

}