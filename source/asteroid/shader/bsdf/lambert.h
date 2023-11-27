#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "asteroid/shader/struct.h"
#include "asteroid/shader/sampling.h"
#include "asteroid/shader/util.h"
#include "asteroid/renderer/scene_struct.h"

namespace Asteroid
{

__device__ glm::vec3 evalLambert(const glm::vec3& v, const glm::vec3& l, const Intersection& its, const Material& mtl)
{
    return mtl.baseColor / glm::pi<float>();
}

__device__ void sampleLambert(const glm::vec3& v,
                              const Intersection& its,
                              const Material& mtl,
                              LCG<16>& rng,
                              ScatterSample& bsdfSample)
{
    auto l = cosineSampleSemiSphere(rng);
    auto transform = onb(its.normal);
    bsdfSample.l = glm::normalize(transform * l);

    bsdfSample.pdf = glm::dot(bsdfSample.l, its.normal) / glm::pi<float>();

    bsdfSample.f = evalLambert(v, bsdfSample.l, its, mtl);
}

}