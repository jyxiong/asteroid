#pragma once

#include "glm/glm.hpp"
#include "asteroid/renderer/ray.h"

namespace Asteroid
{
struct PathSegment {
    Ray ray;
    glm::vec3 color; // accumulated light
    glm::vec3 throughput;
    int pixelIndex;
    int remainingBounces;
};
}
