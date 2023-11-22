#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "asteroid/shader/struct.h"
#include "asteroid/shader/sampling.h"
#include "asteroid/shader/util.h"

namespace Asteroid
{


__device__ glm::vec3 F_Schlick(const glm::vec3& f0, const glm::vec3& f90, float VdotH)
{
    return f0 + (f90 - f0) * glm::pow(glm::clamp(1.f - VdotH, 0.f, 1.f), 5.f);
}

__device__ float V_GGX(float NdotL, float NdotV, float alpha)
{
    auto alpha2 = alpha * alpha;

    auto GGXV = NdotL * glm::sqrt(NdotV * NdotV * (1.f - alpha2) + alpha2);
    auto GGXL = NdotV * glm::sqrt(NdotL * NdotL * (1.f - alpha2) + alpha2);

    auto GGX = GGXV + GGXL;
    return GGX > 0.f ? 0.5f / GGX : 0.f;
}

__device__ float D_GGX(float NdotH, float alpha)
{
    auto alpha2 = alpha * alpha;
    auto f = NdotH * NdotH * (alpha2 - 1.f) + 1.f;
    return alpha2 / (glm::pi<float>() * f * f);
}

__device__ glm::vec3 evalGltf(const glm::vec3& v,
                              const glm::vec3& l,
                              const Intersection& its,
                              const Material& mat)
{
    auto alpha = mat.roughness * mat.roughness;
    auto f0 = glm::mix(glm::vec3(0.04f), mat.baseColor, mat.metallic);
    auto f90 = glm::vec3(1.f);

    auto diffuseColor = glm::mix(mat.baseColor, glm::vec3(0), mat.metallic);

    auto h = glm::normalize(v + l);

    auto NdotL = glm::clamp(glm::dot(its.normal, l), 0.f, 1.f);
    auto NdotV = glm::clamp(glm::dot(its.normal, v), 0.f, 1.f);
    auto NdotH = glm::clamp(glm::dot(its.normal, h), 0.f, 1.f);
    auto VdotH = glm::clamp(glm::dot(v, h), 0.f, 1.f);

    auto F = F_Schlick(f0, f90, VdotH);
    auto V = V_GGX(NdotL, NdotV, alpha);
    auto D = D_GGX(NdotH, glm::max(0.001f, alpha));

    auto diffuse = (1.f - F) * diffuseColor / glm::pi<float>();
    auto specular = F * V * D;

    return diffuse + specular;
}

__device__ void sampleGltf(const glm::vec3& v,
                           const Intersection& its,
                           const Material& mtl,
                           LCG<16>& rng,
                           BsdfSample& bsdfSample)
{
    auto diffuseRatio = 0.5f * (1.f - mtl.metallic);

    auto f0 = glm::mix(glm::vec3(0.04f), mtl.baseColor, mtl.metallic);

    if (rng.rand1() < diffuseRatio) {
        bsdfSample.l = cosineSampleSemiSphere(rng);
        bsdfSample.pdf = glm::dot(bsdfSample.l, its.normal) / glm::pi<float>();
    } else {
        auto h = ggxSampleSemiSphere(mtl.roughness, rng);
        bsdfSample.l = glm::reflect(-v, h);
        bsdfSample.pdf =
            D_GGX(glm::dot(its.normal, h), mtl.roughness) * glm::dot(its.normal, h) / (4.f * glm::dot(v, h));
    }

    auto transform = onb(its.normal);
    bsdfSample.l = glm::normalize(transform * bsdfSample.l);
    bsdfSample.f = evalGltf(v, bsdfSample.l, its, mtl);
}

} // namespace Asteroid
