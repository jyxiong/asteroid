#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/kernel/trace_ray.h"

namespace Asteroid
{

__device__ void directLight()
{

}

__device__ void pathTrace(const SceneView& scene, const RenderState& state, PathSegment& path)
{
    for (int i = 0; i < state.maxDepth; ++i)
    {
        Intersection its{};
        if (intersect(scene, path.ray, its))
        {
            closestHit(scene, its, path);
        }
        else
        {
            miss(scene, its, path);
        }

        if (path.stop)
        {
            break;
        }
    }
}

__device__ glm::vec3 samplePixel(const SceneView& scene, const Camera& camera, const RenderState& state, const glm::ivec2& coord)
{
    auto uv = (glm::vec2(coord) + 0.5f) * 2.f / glm::vec2(state.size) - 1.f;

    auto offsetX = uv.x * camera.tanHalfFov * camera.aspectRatio * camera.right;
    auto offsetY = uv.y * camera.tanHalfFov * camera.up;

    PathSegment path{};
    path.color = glm::vec3(0);
    path.throughput = glm::vec3(1);
    path.ray.direction = glm::normalize(camera.direction + offsetX + offsetY);
    path.ray.origin = camera.position;
    path.stop = false;

    pathTrace(scene, state, path);

    return path.color;
}

__global__ void renderFrameKernel(const SceneView scene,
                                  const Camera camera,
                                  const RenderState state,
                                  BufferView<glm::vec4> image)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    auto viewport = state.size;

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

}
