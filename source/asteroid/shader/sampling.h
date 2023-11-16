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

__device__ inline glm::vec3 uniformSampleSquare(LCG<16>& rng)
{
    auto sample = rng.rand2();

    return { sample.x * 2.f - 1.f, sample.y * 2.f - 1.f, 0.f };
}

__device__ inline glm::vec3 uniformSampleSphere(const Geometry& geometry, LCG<16>& rng)
{
    auto sample = rng.rand2();
    auto z = 1.f - 2.f * sample.x;
    auto r = sqrt(glm::max(0.f, 1.f - z * z));
    auto phi = 2.f * glm::pi<float>() * sample.y;

    return { r * cos(phi), r * sin(phi), z };
}

} // namespace Asteroid