#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/shader/ray_trace/traversal.h"

namespace Asteroid
{

// TODO: t_min & t_max
__device__ bool traceRay(const SceneView& scene, Ray& ray, Intersection& its)
{
    return traversal(scene, ray, its);
}

} // namespace Asteroid
