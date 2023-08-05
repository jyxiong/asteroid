#pragma once

#include "glm/glm.hpp"

namespace Asteroid
{
struct LaunchParams
{
    struct
    {
        unsigned int *colorBuffer;
        glm::ivec2 size;
    } frame;

    struct
    {
        glm::vec3 position;
        glm::vec3 direction;
        glm::vec3 horizontal;
        glm::vec3 vertical;
    } camera;

    OptixTraversableHandle traversable;
};
}