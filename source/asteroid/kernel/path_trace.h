#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/kernel/trace_ray.h"
#include "asteroid/renderer/random.h"

namespace Asteroid
{

__device__ void directLight()
{

}

__device__ void traceRay(const SceneView& scene, PathSegment& path)
{
    Intersection its{};
    if (intersect(scene, path.ray, its))
    {
        closestHit(scene, its, path);
    } else
    {
        miss(scene, its, path);
    }
}

__device__ glm::vec3 samplePixel(const SceneView& scene,
                                 const Camera& camera,
                                 const RenderState& state,
                                 const glm::vec2& coord)
{
    auto pixelColor = glm::vec3(0);

    auto seed = tea<16>(state.size.x * coord.y + coord.x, state.frame * state.maxSamples);

    for (int i = 0; i < state.maxSamples; ++i)
    {
        auto uv = (glm::vec2(coord) + 0.5f) * 2.f / glm::vec2(state.size) - 1.f;

        auto offsetX = uv.x * camera.tanHalfFov * camera.aspectRatio * camera.right;
        auto offsetY = uv.y * camera.tanHalfFov * camera.up;

        for (int i = 0; i < state.maxDepth; ++i)
        {
            PathSegment path{};
            path.color = glm::vec3(0);
            path.throughput = glm::vec3(1);
            path.ray.direction = glm::normalize(camera.direction + offsetX + offsetY);
            path.ray.origin = camera.position;
            path.stop = false;
            path.seed = seed;

            traceRay(scene, path);



            // TODO: if first bounce, store information for denoising
            if (i == 0)
            {

            }

            if (path.stop)
            {
                break;
            }
        }

        pixelColor += path.color;
    }
    pixelColor /= float(state.maxSamples);

    return pixelColor;
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

    auto pixelCoord = glm::vec2(x, y);
    auto pixelColor = samplePixel(scene, camera, state, pixelCoord);

    auto pixelIndex = y * viewport.x + x;
    auto oldColor = glm::vec3(image[pixelIndex]);
    image[pixelIndex] = glm::vec4(glm::mix(oldColor, pixelColor, 1.f / float(state.frame + 1)), 1.f);
}

}
