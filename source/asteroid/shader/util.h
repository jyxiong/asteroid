#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

namespace Asteroid
{

__device__ inline glm::mat3 onb(const glm::vec3& w)
{
    auto sign = std::copysignf(1.0f, w.z);
    auto a = -1.0f / (sign + w.z);
    auto b = w.x * w.y * a;
    auto u = glm::vec3(1.0f + sign * w.x * w.x * a, sign * b, -sign * w.x);
    auto v = glm::vec3(b, sign + w.y * w.y * a, -w.y);
    return { u, v, w };
}

} // namespace Asteroid