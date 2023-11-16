#pragma once

#include <cuda_runtime.h>
#include "asteroid/shader/struct.h"
#include "asteroid/shader/direct_light.h"
#include "asteroid/shader/bsdf/lambert.h"

namespace Asteroid
{
// TODO: trace ray and get closest hit
__device__ void closestHit(const SceneView& scene, const Intersection& its, PathSegment& path)
{
    auto& material = scene.deviceMaterials[its.materialIndex];

    // emission
    if (material.emission.x > 0.f || material.emission.y > 0.f || material.emission.z > 0.f)
    {
        if (path.depth == 0)
        {
            path.radiance += material.emission * path.throughput;
        }
        path.stop = true;
        return;
    }

    // direct light
    path.radiance += directLight(scene, its, material, path.rng) * path.throughput;

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

} // namespace Asteroid