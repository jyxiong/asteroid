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

    return {r * cos(theta), r * sin(theta), sqrt(1 - sample.x)};
}

} // namespace Asteroid