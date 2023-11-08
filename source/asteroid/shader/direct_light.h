#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/shader/struct.h"
#include "asteroid/shader/sampling.h"
#include "asteroid/shader/ray_trace/trace_ray.h"

namespace Asteroid
{

__device__ inline glm::vec3 directLight(const SceneView& scene, const Intersection& its, const Material& mat, LCG<16>& rng)
{
    auto& lights = scene.deviceAreaLights;
    auto index = size_t(rng.rand1() * float(lights.size()));

    LightSample lightSample{};
    uniformSampleOneLight(lights[index], rng, lightSample);

    auto lightDir = glm::normalize(lightSample.position - its.position);
    // 光源必须正对着色点
    if (glm::dot(its.normal, lightSample.normal) > 0.0f)
    {
        return glm::vec3(0);
    }

    Ray shadowRay{};
    shadowRay.origin = its.position + its.normal * 0.0001f;
    shadowRay.direction = lightDir;
    // TODO: any hit test
    Intersection shadowIts{};
    if (traceRay(scene, shadowRay, shadowIts))
    {
        return glm::vec3(0);
    }


    return { };
}

} // namespace Asteroid