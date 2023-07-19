#pragma once

#include "glm/glm.hpp"

namespace Asteroid
{

    struct Intersection
    {
        float t;
        glm::vec3 normal;
        glm::vec3 position;
        int materialId;
    };

}
