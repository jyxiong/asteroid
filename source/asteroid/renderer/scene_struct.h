#pragma once

#include <string>
#include <vector>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Asteroid
{

struct RenderState
{
    unsigned int frame{ 0 };
    unsigned int maxDepth{ 1 };
    unsigned int maxSamples{ 1 };
    glm::ivec2 size{};
};

struct Camera
{
    glm::vec3 position{ 0, 0, 60 };
    glm::vec3 direction{ 0, 0, -1 };
    glm::vec3 up{ 0, 1, 0 };

    float verticalFov{ 45.f };
    float focalDistance{ 0.f };

    glm::vec3 right;
    float tanHalfFov;
    float aspectRatio;

    glm::vec2 lastMousePosition{ 0.0f, 0.0f };
};

enum class MaterialType
{
    Lambert,
    Metal,
    Dielectric,
    Emission
};

struct Material
{
    MaterialType type{};
    glm::vec3 baseColor{1.0f };
    glm::vec3 emission{ 0.0f };

    float roughness = 1.0f;
    float metallic = 0.0f;
};

enum class GeometryType
{
    Disk,
    Square, // -1 < x < 1, -1 < y < 1, z = 0
    Sphere, // x*x + y*y + z*z = 1
    Cube,   // -1 < x < 1, -1 < y < 1, -1 < z < 1
    AABB,
    Mesh
};

struct Geometry
{
    GeometryType type;
    int materialIndex;
    glm::vec3 translation{ 0 };
    glm::vec3 rotation{ 0 };
    glm::vec3 scale{ 1 };
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 inverseTranspose;

    void updateTransform()
    {
        transform = glm::translate(glm::mat4(1.0f), translation)
            * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.x), glm::vec3(1, 0, 0))
            * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.y), glm::vec3(0, 1, 0))
            * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.z), glm::vec3(0, 0, 1))
            * glm::scale(glm::mat4(1.0f), scale);

        inverseTransform = glm::inverse(transform);
        inverseTranspose = glm::transpose(glm::inverse(transform));
    }
};

struct AreaLight
{
    bool enabled{ false };
    bool twoSided{ false };
    size_t geometryId{};
};

} // namespace Asteroid
