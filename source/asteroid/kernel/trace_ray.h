#pragma once

#include <cuda_runtime.h>
#include "asteroid/kernel/struct.h"
#include "asteroid/kernel/intersect.h"
#include "asteroid/kernel/interact.h"

namespace Asteroid
{
__device__ bool intersect(const SceneView& scene, const Ray& ray, Intersection& its)
{
    auto t = std::numeric_limits<float>::max();
    int geometryID = -1;

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
            geometryID = static_cast<int>(i);
        }
    }

    return geometryID >= 0;
}

__device__ void closestHit(const SceneView& scene, const Intersection& its, PathSegment& path)
{
    auto& material = scene.deviceMaterials[its.materialIndex];

    if (material.emittance > 0.0f)
    {
        path.color += (material.albedo * material.emittance) * path.throughput;
        path.stop = true;
    } else
    {
        scatterRay(its, material, path);
    }

    // TODO: russian roulette
}

__device__ void miss(const SceneView& scene, const Intersection& its, PathSegment& path)
{
    // TODO: environment light


    path.stop = true;
}

__device__ void anyHit()
{

}

} // namespace Asteroid