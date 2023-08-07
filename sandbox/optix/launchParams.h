#pragma once

#include "glm/glm.hpp"

namespace Asteroid
{
struct TriangleMeshSBTData
{
    glm::vec3 color;
    glm::vec3 *vertex;
    glm::ivec3 *index;
};

struct LaunchParams
{
    struct
    {
        glm::u8vec4 *colorBuffer;
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