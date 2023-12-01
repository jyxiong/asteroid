#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "asteroid/shader/struct.h"
#include "asteroid/shader/sampling.h"
#include "asteroid/shader/util.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/shader/bsdf/fresnel.h"

namespace Asteroid::Dielectric
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
    auto f = fresnel(v, its.normal, mtl.ior);

    if (rng.rand1() < f) {
        bsdfSample.l = glm::reflect(-v, its.normal);
    } else {
        bsdfSample.l = glm::refract(-v, its.normal, 1.f / mtl.ior);
    }

    bsdfSample.pdf = 1.f;
    bsdfSample.f = glm::vec3(1);
}

} // namespace Asteroid
