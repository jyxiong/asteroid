#pragma once

#include <curand_kernel.h>
#include "glm/glm.hpp"
#include "random.h"

namespace Asteroid
{

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct PathSegment
{
    int depth;
    Ray ray;
    glm::vec3 radiance;
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
    int geometryIndex;
};

struct LightSample
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 emission;
    float pdf;
};

struct BsdfSample
{
    glm::vec3 l;
    glm::vec3 f;
    float pdf;
};

} // namespace Asteroid