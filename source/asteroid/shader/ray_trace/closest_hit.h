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
    if (material.emission.x > 0.f || material.emission.y > 0.f || material.emission.z > 0.f) {
        path.radiance += material.emission * path.throughput;
        path.stop = true;
        return;
    }

    // direct light
    path.radiance += directLight(scene, path.ray, its, material, path.rng) * path.throughput;

    // indirect light
    BsdfSample bsdfSample{};
    sampleGltf(-path.ray.direction, its.normal, material, path.rng, bsdfSample);

    if (bsdfSample.pdf < 0.f) {
        path.stop = true;
        return;
    }

    auto NdotL = glm::dot(bsdfSample.l, its.normal);
    if (NdotL < 0.f) {
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