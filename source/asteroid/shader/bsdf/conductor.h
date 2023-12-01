#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "asteroid/shader/struct.h"
#include "asteroid/shader/sampling.h"
#include "asteroid/shader/util.h"
#include "asteroid/renderer/scene_struct.h"

namespace Asteroid::Conductor
{
__device__ glm::vec3 eval(const glm::vec3& v, const glm::vec3& l, const Intersection& its, const Material& mtl)
{
    return glm::vec3(0);
}

__device__ void sample(const glm::vec3& v,
                       const Intersection& its,
                       const Material& mtl,
                       LCG<16>& rng,
                       ScatterSample& bsdfSample)
{
    bsdfSample.l = glm::reflect(-v, its.normal);
    bsdfSample.pdf = 1.f;
    bsdfSample.f = glm::vec3(1);
}

} // namespace Asteroid