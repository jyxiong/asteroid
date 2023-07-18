#pragma once

#include "glm/glm.hpp"

namespace Asteroid
{

    struct Intersection
    {
        float t;
        glm::vec3 normal;
        glm::vec3 position;

        glm::vec3 albedo;
        int materialId;
    };

    struct HitPayload
    {
        float HitDistance;
        glm::vec3 WorldPosition;
        glm::vec3 WorldNormal;

        int ObjectIndex;
    };

}
