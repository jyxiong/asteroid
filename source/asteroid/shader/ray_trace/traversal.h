#pragma once

#include <cuda_runtime.h>
#include "asteroid/shader/struct.h"
#include "asteroid/shader/ray_trace/intersection.h"
#include "asteroid/shader/ray_trace/any_hit.h"

namespace Asteroid
{
__device__ bool traversal(const SceneView& scene, const Ray& ray, Intersection& its)
{
    auto t = std::numeric_limits<float>::max();
    its.geometryIndex = -1;

    for (size_t i = 0; i < scene.deviceGeometries.size(); ++i)
    {
        const auto& geometry = scene.deviceGeometries[i];

        if (geometry.type == GeometryType::Sphere)
        {
            if (!intersectSphere(geometry, ray, its))
            {
                continue;
            }
        } else if (geometry.type == GeometryType::Cube)
        {
            if (!intersectCube(geometry, ray, its))
            {
                continue;
            }
        }

        if (its.t < t && its.t > 0)
        {
            t = its.t;
            its.geometryIndex = static_cast<int>(i);

            // TODO: any hit test
            anyHit();
        }

    }

    return its.geometryIndex >= 0;
}

} // namespace Asteroid