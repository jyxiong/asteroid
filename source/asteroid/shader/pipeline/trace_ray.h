#pragma once

#include <cuda_runtime.h>
#include "asteroid/shader/struct.h"
#include "intersect.h"
#include "asteroid/shader/directLight.h"
#include "asteroid/shader/bsdf/lambert.h"

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

    // emittance
    path.radiance += material.emittance * path.throughput;
        
    // direct light
    path.radiance += directLight(its, material) * path.throughput;

    // indirect light
    BsdfSample bsdfSample{};
    lambertSample(-path.ray.direction, its, material, path.rng, bsdfSample);

    if (bsdfSample.pdf < 0.f)
    {
        path.stop = true;
        return;
    }

    path.throughput *= bsdfSample.f * glm::dot(bsdfSample.l, its.normal) / bsdfSample.pdf;

    // next ray
    path.ray.origin = its.position + its.normal * 0.0001f;
    path.ray.direction = bsdfSample.l;

    // TODO: russian roulette
}

__device__ void miss(const SceneView& scene, const Intersection& its, PathSegment& path)
{
    // TODO: environment light
    path.stop = true;
}

[[maybe_unused]] [[maybe_unused]] __device__ void anyHit()
{

}

} // namespace Asteroid