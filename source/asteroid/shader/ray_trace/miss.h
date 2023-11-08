#pragma once

#include <cuda_runtime.h>
#include "asteroid/shader/struct.h"
#include "asteroid/shader/direct_light.h"
#include "asteroid/shader/bsdf/lambert.h"

namespace Asteroid
{
__device__ void miss(const SceneView& scene, const Intersection& its, PathSegment& path)
{
    // TODO: environment light
    path.stop = true;
}

} // namespace Asteroid