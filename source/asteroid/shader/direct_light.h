#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/shader/struct.h"
#include "asteroid/shader/sampling.h"
#include "asteroid/shader/bsdf/lambert.h"
#include "asteroid/shader/ray_trace/trace_ray.h"

namespace Asteroid
{

__device__ inline void uniformSampleOneLight(const Geometry& geometry, const Material& material, LCG<16>& rng, LightSample& lightSample)
{
    if (geometry.type == GeometryType::Sphere)
    {
        auto point = uniformSampleSphere(geometry, rng);
        lightSample.position = glm::vec3(geometry.transform * glm::vec4(point, 1.f));
        lightSample.normal = glm::normalize(glm::vec3(geometry.transform * glm::vec4(point, 0.f)));
        lightSample.emission = material.emission;
        lightSample.pdf = 1.f / (4.f * glm::pi<float>());
    }
    else if (geometry.type == GeometryType::Cube)
    {
    }
}

__device__ inline glm::vec3 directLight(const SceneView& scene, const Intersection& its, const Material& mat, LCG<16>& rng)
{
    auto& lights = scene.deviceAreaLights;
    auto lightIndex = size_t(rng.rand1() * float(lights.size()));
    auto geometryIndex = lights[lightIndex].geometryId;
    auto lightGeometry = scene.deviceGeometries[geometryIndex];
    auto lightMaterial = scene.deviceMaterials[lightGeometry.materialIndex];

    LightSample lightSample{};
    uniformSampleOneLight(lightGeometry, lightMaterial, rng, lightSample);

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
        if (shadowIts.geometryIndex != geometryIndex)
        {
            return glm::vec3(0);
        }
    }

    auto f = lambertEval(-its.normal, lightDir, its, mat);

    return lightSample.emission * glm::dot(its.normal, lightDir) * f / lightSample.pdf;
}

} // namespace Asteroid