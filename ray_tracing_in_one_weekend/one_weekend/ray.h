#pragma once

#include "glm/glm.hpp"

class Ray
{
public:
    glm::vec3 origin;
    glm::vec3 direction;

public:

    __device__
    Ray() {}

    __device__
    Ray(const glm::vec3 &origin, const glm::vec3 &direction)
            : origin(origin), direction(direction) {}

    __device__
    glm::vec3 operator()(float t) const
    {
        return origin + t * direction;
    }
};