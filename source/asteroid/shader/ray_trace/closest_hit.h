#pragma once

#include <cuda_runtime.h>
#include "asteroid/shader/struct.h"
#include "direct_light.h"
#include "asteroid/shader/bsdf/diffuse.h"

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
    ScatterSample scatterSample{};
    sampleGltf(-path.ray.direction, its.normal, material, path.rng, scatterSample);

    if (scatterSample.pdf < 0.f) {
        path.stop = true;
        return;
    }

//    auto NdotL = glm::dot(scatterSample.l, its.normal);
//    if (NdotL < 0.f) {
//        path.stop = true;
//        return;
//    }

    path.throughput *= scatterSample.f / scatterSample.pdf;

    // next ray
    path.ray.origin = its.position + its.normal * 0.0001f;
    path.ray.direction = scatterSample.l;

    // TODO: russian roulette
}

} // namespace Asteroid