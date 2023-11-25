#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/shader/random.h"
#include "asteroid/shader/struct.h"
#include "asteroid/shader/ray_trace/path_trace.h"


namespace Asteroid
{

__device__ glm::vec3 samplePixel(const SceneView scene,
                                 const Camera camera,
                                 const RenderState state,
                                 const glm::ivec2& coord,
                                 LCG<16>& rng)
{
    // sample ray in one pixel
    auto pixelCenter = glm::vec2(coord) + rng.rand2();
    auto uv = pixelCenter / glm::vec2(state.size) * 2.f - 1.f;

    auto offsetX = uv.x * camera.tanHalfFov * camera.aspectRatio * camera.right;
    auto offsetY = uv.y * camera.tanHalfFov * camera.up;
    auto direction = glm::normalize(camera.direction + offsetX + offsetY);
    // TODO: depth of field
    auto origin = camera.position;

    PathSegment path{};
    path.radiance = glm::vec3(0);
    path.throughput = glm::vec3(1);
    path.ray.direction = direction;
    path.ray.origin = origin;
    path.stop = false;
    path.rng = rng;
    path.depth = 0;

    pathTrace(scene, state, path);

    return path.radiance;
}

} // namespace Asteroid
