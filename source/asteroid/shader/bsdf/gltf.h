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

__device__ void evalGltf(const glm::vec3& v,
                         const glm::vec3& n,
                         const Material& mat,
                         BsdfSample& bsdfSample)
{
    auto alpha = mat.roughness * mat.roughness;
    auto f0 = glm::mix(glm::vec3(0.04f), mat.baseColor, mat.metallic);
    auto f90 = glm::vec3(1.f);

    auto diffuseColor = glm::mix(mat.baseColor, glm::vec3(0), mat.metallic);

    auto h = glm::normalize(v + bsdfSample.l);

    auto NdotL = glm::dot(n, bsdfSample.l);
    if (NdotL < 0.f) {
        bsdfSample.pdf = 0.f;
        bsdfSample.f = glm::vec3(0.f);
        return;
    }
    NdotL = glm::clamp(NdotL, 0.001f, 1.f);
    auto NdotV = glm::dot(n, v);
    NdotV = glm::clamp(glm::abs(NdotV), 0.001f, 1.f);

    auto NdotH = glm::clamp(glm::dot(n, h), 0.f, 1.f);
    auto VdotH = glm::clamp(glm::dot(v, h), 0.f, 1.f);

    auto F = F_Schlick(f0, f90, VdotH);
    auto V = V_GGX(NdotL, NdotV, alpha);
    auto D = D_GGX(NdotH, glm::max(0.001f, alpha));

    auto diffuse = (1.f - F) * diffuseColor / glm::pi<float>();
    auto specular = F * V * D;

    bsdfSample.f = diffuse + specular;

    auto diffuseRatio = 0.f;//0.5f * (1.f - mat.metallic);
    auto diffusePdf = NdotL / glm::pi<float>();
    auto specularPdf = D * NdotH / (4.f * VdotH);

    bsdfSample.pdf = glm::mix(specularPdf, diffusePdf, diffuseRatio);
}

__device__ void sampleGltf(const glm::vec3& v,
                           const glm::vec3& n,
                           const Material& mtl,
                           LCG<16>& rng,
                           BsdfSample& bsdfSample)
{
    auto transform = onb(n);

    auto diffuseRatio = 0.f;//0.5f * (1.f - mtl.metallic);
    if (rng.rand1() < diffuseRatio) {
        auto l = cosineSampleSemiSphere(rng);
        bsdfSample.l = glm::normalize(transform * l);
    } else {
        auto h = ggxSampleSemiSphere(mtl.roughness, rng);
        h = glm::normalize(transform * h);
        bsdfSample.l = glm::normalize(glm::reflect(-v, h));
    }

    evalGltf(v, n, mtl, bsdfSample);
}

} // namespace Asteroid
