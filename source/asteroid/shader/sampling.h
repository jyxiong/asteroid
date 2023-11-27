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
    auto phi = 2 * glm::pi<float>() * sample.y;

    return {r * cos(phi), r * sin(phi), sqrt(1 - sample.x)};
}

__device__ inline glm::vec3 uniformSampleSemiSphere(LCG<16>& rng)
{
    auto sample = rng.rand2();

    auto r = sqrt(1 - sample.x * sample.x);
    auto phi = 2 * glm::pi<float>() * sample.y;

    return {r * cos(phi), r * sin(phi), sample.x};
}

__device__ inline glm::vec3 uniformSampleSquare(LCG<16>& rng)
{
    auto sample = rng.rand2();

    return {sample.x * 2.f - 1.f, sample.y * 2.f - 1.f, 0.f};
}

__device__ inline glm::vec3 uniformSampleSphere(LCG<16>& rng)
{
    auto sample = rng.rand2();
    auto z = 1.f - 2.f * sample.x;
    auto r = sqrt(glm::max(0.f, 1.f - z * z));
    auto phi = 2.f * glm::pi<float>() * sample.y;

    return {r * cos(phi), r * sin(phi), z};
}

__device__ inline glm::vec3 ggxSampleSemiSphere(float roughness, LCG<16>& rng)
{
    auto sample = rng.rand2();

    auto phi = 2.f * glm::pi<float>() * sample.x;
    auto cosTheta = glm::sqrt((1.f - sample.y) / (1.f + (roughness * roughness - 1.f) * sample.y));
    auto sinTheta = glm::sqrt(1.f - cosTheta * cosTheta);

    return {sinTheta * glm::cos(phi), sinTheta * glm::sin(phi), cosTheta};
}

__device__ inline float powerHeuristic(float a, float b)
{
    auto t = a * a;
    return t / (b * b + t);
}

} // namespace Asteroid