#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

struct Ray
{
    glm::vec3 Origin;

    glm::vec3 Direction;

    __device__ glm::vec3 operator()(float t) const
    {
        return Origin + t * Direction;
    }
};