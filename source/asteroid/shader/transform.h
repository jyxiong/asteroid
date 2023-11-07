#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Asteroid
{

struct Transform
{
    glm::vec3 translation{ 0 };
    glm::vec3 rotation{ 0 };
    glm::vec3 scale{ 1 };

    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 inverseTranspose;

    void update()
    {
        transform = glm::translate(glm::mat4(1.0f), translation)
            * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.x), glm::vec3(1, 0, 0))
            * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.y), glm::vec3(0, 1, 0))
            * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.z), glm::vec3(0, 0, 1))
            * glm::scale(glm::mat4(1.0f), scale);

        inverseTransform = glm::inverse(transform);
        inverseTranspose = glm::transpose(glm::inverse(transform));
    }

    __device__ inline glm::vec3 xformPoint(const glm::vec3& point) const
    {
        return { transform * glm::vec4(point, 1.0f) };
    }

    __device__ inline glm::vec3 inverseXformPoint(const glm::vec3& point) const
    {
        return { inverseTransform * glm::vec4(point, 1.0f) };
    }

    __device__ inline glm::vec3 xformVector(const glm::vec3& vector) const
    {
        return { transform * glm::vec4(vector, 0.0f) };
    }

    __device__ inline glm::vec3 inverseXformVector(const glm::vec3& vector) const
    {
        return { inverseTransform * glm::vec4(vector, 0.0f) };
    }

    __device__ inline glm::vec3 xformNormal(const glm::vec3& normal) const
    {
        return { inverseTranspose * glm::vec4(normal, 0.0f) };
    }
};

} // namespace Asteroid