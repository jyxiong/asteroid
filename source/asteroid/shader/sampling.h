#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "random.h"

namespace Asteroid
{

__device__ inline glm::vec3 cosineSampleSemiSphere(LCG<16>& rng)
{
    auto sample = rng.rand2();

    auto r = sqrt(sample.x);
    auto theta = 2 * glm::pi<float>() * sample.y;

    return { r * cos(theta), r * sin(theta), sqrt(1 - sample.x) };
}

__device__ inline void uniformSampleSquare(const AreaLight& light, LCG<16>& rng, LightSample& lightSample)
{
    auto sample = rng.rand2() * 2.f - 1.f;
    lightSample.position = glm::vec3(light.transform * glm::vec4(sample, 0.f, 0.f));
    lightSample.emission = light.emission;
    lightSample.normal = glm::normalize(glm::vec3(light.transform * glm::vec4(0.f, 0.f, 1.f, 0.f)));
    lightSample.pdf = 1.f / light.area;
}

__device__ inline void uniformSampleOneLight(const AreaLight& light, LCG<16>& rng, LightSample& lightSample)
{
    if (light.type == LightType::Square)
    {
        uniformSampleSquare(light, rng, lightSample);
    }
}

} // namespace Asteroid