#pragma once

#include <cuda_runtime.h>
#include "asteroid/shader/struct.h"
#include "direct_light.h"
#include "asteroid/shader/bsdf/lambert.h"

namespace Asteroid
{
__device__ void miss(const SceneView& scene, const Intersection& its, PathSegment& path)
{
    // TODO: environment light
//    path.radiance = glm::vec3(1, 0, 0);
    path.stop = true;
}

} // namespace Asteroid