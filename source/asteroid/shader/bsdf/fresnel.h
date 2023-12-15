#include <cuda_runtime.h>

#include "glm/glm.hpp"

namespace Asteroid
{

__device__ float fresnel(const glm::vec3& v, const glm::vec3& n, float ior)
{
    auto cosThetaI = glm::dot(v, n);

    auto etaI = 1.0f;
    auto etaT = ior;

    if (cosThetaI > 0)
    {
        auto tmp = etaI;
        etaI = etaT;
        etaT = tmp;
    }

    auto sinThetaI = glm::sqrt(glm::max(0.0f, 1.0f - cosThetaI * cosThetaI));
    auto sinThetaT = etaI / etaT * sinThetaI;

    if (sinThetaT >= 1.0f)
    {
        return 1.f;
    }

    auto cosThetaT = glm::sqrt(glm::max(0.0f, 1.0f - sinThetaT * sinThetaT));

    auto rParl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    auto rPerp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));

    return (rParl * rParl + rPerp * rPerp) / 2.0f;

}

} // namespace Asteroid