#pragma once

#include <cuda_runtime.h>
#include "asteroid/shader/struct.h"
#include "asteroid/shader/ray_trace/intersection.h"
#include "asteroid/shader/ray_trace/any_hit.h"

namespace Asteroid
{
__device__ bool traversal(const SceneView& scene, const Ray& ray, Intersection& its)
{
    its.t = std::numeric_limits<float>::max();
    its.geometryIndex = -1;

    Intersection itsTemp{};
    for (size_t i = 0; i < scene.deviceGeometries.size(); ++i)
    {
        const auto& geometry = scene.deviceGeometries[i];

        if (geometry.type == GeometryType::Sphere)
        {
            if (!intersectSphere(geometry, ray, itsTemp))
            {
                continue;
            }
        } else if (geometry.type == GeometryType::Cube)
        {
            if (!intersectCube(geometry, ray, itsTemp))
            {
                continue;
            }
        }

        if (itsTemp.t < its.t && itsTemp.t > 0)
        {
            its.t = itsTemp.t;
            its.normal = itsTemp.normal;
            its.position = itsTemp.position;
            its.geometryIndex = static_cast<int>(i);
            its.materialIndex = geometry.materialIndex;
        }

    }

    return its.geometryIndex >= 0;
}

} // namespace Asteroid