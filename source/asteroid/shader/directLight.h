#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/shader/struct.h"

namespace Asteroid
{

__device__ inline glm::vec3 directLight(const Intersection& its, const Material& mat)
{

    auto lightDir = glm::normalize(glm::vec3(-1, -1, -1));
    auto lightIntensity = glm::max(glm::dot(its.normal, -lightDir), 0.0f);

    auto color = mat.albedo * lightIntensity;

    // TODO: shadow ray with any hit

    return color;
}

} // namespace Asteroid