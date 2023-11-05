#pragma once

#include "glm/glm.hpp"

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
    unsigned int seed;
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