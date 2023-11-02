#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "glm/glm.hpp"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/kernel/intersect.h"
#include "asteroid/kernel/path_trace.h"

namespace Asteroid
{
__global__ void renderFrame(const SceneView scene,
                            const Camera camera,
                            const RenderState state,
                            BufferView<glm::vec4> image)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    auto viewport = camera.viewport;

    if (x >= viewport.x && y >= viewport.y)
        return;

    
    auto pixelColor = glm::vec3(0);
    for (int i = 0; i < state.maxSamples; ++i)
    {
        pixelColor += samplePixel(scene, camera, state, { x, y });
    }
    pixelColor /= float(state.maxSamples);

    auto pixelIndex = y * viewport.x + x;
    auto oldColor = glm::vec3(image[pixelIndex]);
    image[pixelIndex] = glm::vec4(glm::mix(oldColor, pixelColor, 1.f / float(state.frame + 1)), 1.f);
}

__device__
void scatterRay(PathSegment& pathSegment, const Intersection& its, const Material& mat)
{

    auto lightDir = glm::normalize(glm::vec3(-1, -1, -1));
    auto lightIntensity = glm::max(glm::dot(its.normal, -lightDir), 0.0f);

    auto color = mat.albedo * lightIntensity;
    pathSegment.color += color * pathSegment.throughput;

    pathSegment.throughput *= 0.5f;

    pathSegment.ray.origin = its.position + its.normal * 0.0001f;
    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, its.normal + mat.roughness);

    pathSegment.remainingBounces--;
}

__global__ void generatePathSegment(const Camera camera, unsigned int traceDepth, BufferView<PathSegment> pathSegments)
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

    auto uv = glm::vec2(x, y) / glm::vec2(viewport) * 2.f - 1.f;
    auto offsetX = float(uv.x) * camera.tanHalfFov * camera.aspectRatio * camera.right;
    auto offsetY = float(uv.y) * camera.tanHalfFov * camera.up;

    path.ray.direction = glm::normalize(camera.direction + offsetX + offsetY);
    path.ray.origin = camera.position;
}

__global__ void computeIntersection(const SceneView scene, int width, int height,
                                    BufferView<PathSegment> pathSegments,
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
    Intersection its{};
    float hitDistance = std::numeric_limits<float>::max();
    for (size_t i = 0; i < scene.deviceGeometries.size(); ++i)
    {
        const auto& geometry = scene.deviceGeometries[i];

        if (geometry.type == GeometryType::Sphere)
        {
            if (!intersectSphere(geometry, path.ray, its))
            {
                continue;
            }
        } else if (geometry.type == GeometryType::Cube)
        {
            if (!intersectCube(geometry, path.ray, its))
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

__global__ void shading(const SceneView scene, const BufferView<Intersection> its, const glm::ivec2 size,
                        BufferView<PathSegment> pathSegments)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size.x && y >= size.y)
        return;

    if (pathSegments[y * size.x + x].remainingBounces == 0)
    {
        return;
    }

    auto& it = its[y * size.x + x];
    auto& material = scene.deviceMaterials[it.materialIndex];
    auto& path = pathSegments[y * size.x + x];

    if (material.emittance > 0.0f)
    {
        path.color += (material.albedo * material.emittance) * path.throughput;
        path.remainingBounces = 0;
    } else
    {
        scatterRay(path, it, material);
    }
}

__global__ void finalGather(const BufferView<PathSegment> pathSegments,
                            unsigned int frame,
                            int width,
                            int height,
                            BufferView<glm::vec4> image)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width && y >= height)
        return;

    const auto& path = pathSegments[y * width + x];
    auto oldColor = glm::vec3(image[y * width + x]);
    image[y * width + x] = glm::vec4(glm::mix(oldColor, path.color, 1.f / float(frame + 1)), 1.f);
}
}