#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/shader/ray_trace/traversal.h"
#include "asteroid/shader/ray_trace/closest_hit.h"
#include "asteroid/shader/ray_trace/miss.h"

namespace Asteroid
{

// TODO: t_min & t_max
__device__ void pathTrace(const SceneView& scene, const RenderState state, PathSegment& path)
{
    // trace depth
    while (path.depth < state.maxDepth) {
        Intersection its{};
        if (traversal(scene, path.ray, its)) {
            closestHit(scene, its, path);
        } else {
            miss(scene, its, path);
        }

        // TODO: if first bounce, store information for denoising
        if (path.depth == 0) {

        }

        if (path.stop) {
            break;
        }

        path.depth++;
    }
}

} // namespace Asteroid
