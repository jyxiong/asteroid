#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/shader/random.h"
#include "asteroid/shader/struct.h"
#include "asteroid/shader/ray_trace/trace_ray.h"
#include "asteroid/shader/ray_trace/closest_hit.h"
#include "asteroid/shader/ray_trace/miss.h"

namespace Asteroid
{

__global__ void rayGeneration(const SceneView scene,
                              const Camera camera,
                              const RenderState state,
                              BufferView<glm::vec4> image)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    auto& viewport = state.size;

    if (x >= viewport.x && y >= viewport.y)
        return;

    auto pixelIndex = y * viewport.x + x;

    auto rng = LCG<16>(pixelIndex, state.frame);

    auto uv = (glm::vec2(x, y) + rng.rand2()) * 2.f / glm::vec2(state.size) - 1.f;
    auto offsetX = uv.x * camera.tanHalfFov * camera.aspectRatio * camera.right;
    auto offsetY = uv.y * camera.tanHalfFov * camera.up;
    auto direction = glm::normalize(camera.direction + offsetX + offsetY);
    // TODO: depth of field
    auto origin = camera.position;

    auto pixelColor = glm::vec3(0);
    for (int sample_idx = 0; sample_idx < state.maxSamples; ++sample_idx)
    {
        PathSegment path{};
        path.radiance = glm::vec3(0);
        path.throughput = glm::vec3(1);
        path.ray.direction = direction;
        path.ray.origin = origin;
        path.stop = false;
        path.rng = rng;

        for (int depth = 0; depth < state.maxDepth; ++depth)
        {
            Intersection its{};
            if (traceRay(scene, path.ray, its))
            {
                closestHit(scene, its, path);
            } else
            {
                miss(scene, its, path);
            }

            // TODO: if first bounce, store information for denoising
            if (depth == 0)
            {

            }

            if (path.stop)
            {
                break;
            }
        }

        pixelColor += path.radiance;

        rng = path.rng;
    }
    pixelColor /= float(state.maxSamples);

    auto oldColor = glm::vec3(image[pixelIndex]);
    auto newColor = glm::mix(oldColor, pixelColor, 1.f / float(state.frame + 1));

    // TODO: tone mapping
    // TODO: gamma correction

    image[pixelIndex] = glm::vec4(newColor, 1.f);
}

}
