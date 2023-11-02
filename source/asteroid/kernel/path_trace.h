#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/kernel/traceRay.h"

namespace Asteroid
{

__device__ void directLight()
{

}

__device__ glm::vec3 pathTrace(const SceneView& scene, const RenderState& state, const Ray& ray)
{
    auto radiance = glm::vec3(0);

    for (int i = 0; i < state.maxDepth; ++i)
    {
        closestHit();

        if 
    }

    return glm::vec3(0);
}

__device__ glm::vec3 samplePixel(const SceneView& scene, const Camera& camera, const RenderState& state, const glm::ivec2& coord)
{
    auto uv = (glm::vec2(coord) + 0.5f) * 2.f / glm::vec2(state.size) - 1.f;

    auto offsetX = uv.x * camera.tanHalfFov * camera.aspectRatio * camera.right;
    auto offsetY = uv.y * camera.tanHalfFov * camera.up;

    Ray ray{};
    ray.direction = glm::normalize(camera.direction + offsetX + offsetY);
    ray.origin = camera.position;

    auto radiance = pathTrace(scene, state, ray);

    return radiance;
}

}
