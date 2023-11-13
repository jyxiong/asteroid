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

__device__ glm::vec3 rayGeneration(const SceneView scene,
                                   const Camera camera,
                                   const RenderState state,
                                   const glm::ivec2& coord)
{
    auto pixelIndex = coord.y * state.size.x + coord.x;
    auto rng = LCG<16>(pixelIndex, state.frame);

    auto uv = (glm::vec2(coord) + rng.rand2()) * 2.f / glm::vec2(state.size) - 1.f;

    auto offsetX = uv.x * camera.tanHalfFov * camera.aspectRatio * camera.right;
    auto offsetY = uv.y * camera.tanHalfFov * camera.up;
    auto direction = glm::normalize(camera.direction + offsetX + offsetY);
    // TODO: depth of field
    auto origin = camera.position;

    auto pixelColor = glm::vec3(0);

    // sample per pixel in one frame
    for (int sample = 0; sample < state.maxSamples; ++sample)
    {
        PathSegment path{};
        path.radiance = glm::vec3(0);
        path.throughput = glm::vec3(1);
        path.ray.direction = direction;
        path.ray.origin = origin;
        path.stop = false;
        path.rng = rng;

        // trace depth
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

    return pixelColor / float(state.maxSamples);
}

} // namespace Asteroid
