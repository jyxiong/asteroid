#pragma once

#include <curand_kernel.h>
#include "glm/glm.hpp"
#include "asteroid/cuda/random.h"

namespace Asteroid
{

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    glm::vec3 throughput;
    bool stop;
    LCG<16> rng;
};

struct Intersection
{
    float t;
    glm::vec3 normal;
    bool front_face;
    glm::vec3 position;
    int materialIndex;
};

} // namespace Asteroid